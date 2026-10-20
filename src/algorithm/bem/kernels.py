"""CUDA kernels for the Burton-Miller collocation matrix and field evaluation.

Time convention exp(-i w t), Green's function G = exp(i k r) / (4 pi r).
Element integrals use flat triangles with constant density, collocated at
centroids. Quadrature adapts to the distance d between the collocation point
and the element (in units of the element's longest edge):

    d >= 4      one point at the centroid
    2 <= d < 4  7-point degree-5 rule
    smaller d   the element is split into 4^m congruent triangles (m = 1..3),
                each with the 7-point rule

Device arithmetic is single precision (FP64 sin/cos made assembly ~10x slower and
the sphere validation shows no accuracy change); the singular self terms are
computed in double precision separately (``bem.solver``):
the double-layer self term vanishes on a flat element and the hypersingular self
term is a Hadamard finite part.
"""

import cupy as cp

_SOURCE = r"""
#include <cupy/complex.cuh>
#include <cuda_fp16.h>
#define PI 3.14159265358979323846

__device__ const float QB[7][3] = {
    {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
    {0.0597158717, 0.4701420641, 0.4701420641},
    {0.4701420641, 0.0597158717, 0.4701420641},
    {0.4701420641, 0.4701420641, 0.0597158717},
    {0.7974269853, 0.1012865073, 0.1012865073},
    {0.1012865073, 0.7974269853, 0.1012865073},
    {0.1012865073, 0.1012865073, 0.7974269853}};
__device__ const float QW[7] = {
    0.225, 0.1323941527, 0.1323941527, 0.1323941527,
    0.1259391805, 0.1259391805, 0.1259391805};

// acc[0..1] += w dG/dn_y, acc[2..3] += w d2G/dn_x dn_y (if need_h)
__device__ void point_kernel(const float* x, const float* nx, const float* y,
                             const float* ny, float k, float w, int need_h,
                             float* acc) {
    float rx = y[0] - x[0], ry = y[1] - x[1], rz = y[2] - x[2];
    float r = sqrtf(rx * rx + ry * ry + rz * rz);
    float kr = k * r, c, s;
    // kr is only known to float precision anyway; reduce to [-pi, pi] and use the
    // hardware sin/cos (as accurate as sinf/cosf here, several times faster).
    __sincosf(kr - 6.28318530718f * rintf(kr * 0.159154943092f), &s, &c);
    float scale = 1.0 / (4.0 * PI * r * r);
    float gp_r = (-c - s * kr) * scale, gp_i = (c * kr - s) * scale;  // G'
    float dny = (rx * ny[0] + ry * ny[1] + rz * ny[2]) / r;
    acc[0] += w * gp_r * dny;
    acc[1] += w * gp_i * dny;
    if (need_h) {
        float dnx = -(rx * nx[0] + ry * nx[1] + rz * nx[2]) / r;
        float nn = nx[0] * ny[0] + nx[1] * ny[1] + nx[2] * ny[2];
        float a = 2.0 - kr * kr;
        float gpp_r = (c * a + 2.0 * s * kr) * scale / r;  // G''
        float gpp_i = (s * a - 2.0 * c * kr) * scale / r;
        float t1 = dnx * dny, t2 = (-nn - dnx * dny) / r;
        acc[2] += w * (gpp_r * t1 + gp_r * t2);
        acc[3] += w * (gpp_i * t1 + gp_i * t2);
    }
}

__device__ void sub_triangle(const float* x, const float* nx, const float* v,
                             const float* ny, const float* p, const float* q,
                             const float* rr, float weight, float k, int need_h,
                             float* acc) {
    for (int t = 0; t < 7; ++t) {
        float l1 = QB[t][0] * p[0] + QB[t][1] * q[0] + QB[t][2] * rr[0];
        float l2 = QB[t][0] * p[1] + QB[t][1] * q[1] + QB[t][2] * rr[1];
        float y[3];
        for (int d = 0; d < 3; ++d)
            y[d] = v[d] + l1 * (v[3 + d] - v[d]) + l2 * (v[6 + d] - v[d]);
        point_kernel(x, nx, y, ny, k, weight * QW[t], need_h, acc);
    }
}

__device__ void element(const float* x, const float* nx, const float* v,
                        const float* centroid, const float* ny, float area,
                        float hsize, float k, int need_h, float* acc) {
    float dx = centroid[0] - x[0], dy = centroid[1] - x[1], dz = centroid[2] - x[2];
    float d = sqrtf(dx * dx + dy * dy + dz * dz) / hsize;
    if (d >= 4.0) {
        point_kernel(x, nx, centroid, ny, k, area, need_h, acc);
        return;
    }
    int m = d >= 2.0 ? 0 : d >= 1.0 ? 1 : d >= 0.5 ? 2 : 3;
    int s = 1 << m;
    float weight = area / (s * s), inv = 1.0 / s;
    for (int a = 0; a < s; ++a) {
        for (int b = 0; b < s - a; ++b) {
            float p[2] = {a * inv, b * inv}, q[2] = {(a + 1) * inv, b * inv};
            float rr[2] = {a * inv, (b + 1) * inv};
            sub_triangle(x, nx, v, ny, p, q, rr, weight, k, need_h, acc);
            if (a + b <= s - 2) {
                float p2[2] = {(a + 1) * inv, (b + 1) * inv};
                sub_triangle(x, nx, v, ny, q, rr, p2, weight, k, need_h, acc);
            }
        }
    }
}

__device__ void load(const float* verts, const float* cent, const float* nrm, int j,
                     const float* sign, const float* offset, float* v, float* c, float* n) {
    for (int d = 0; d < 9; ++d) v[d] = sign[d % 3] * verts[9 * j + d] + offset[d % 3];
    for (int d = 0; d < 3; ++d) {
        c[d] = sign[d] * cent[3 * j + d] + offset[d];
        n[d] = sign[d] * nrm[3 * j + d];
    }
}

// A_ij = base_ij + [i == j and image 0 included] / 2 - K_ij - alpha H_ij over images
// [first, last) (row i: collocation point, column j: element; base may be null)
extern "C" __global__ void assemble(
    const int n, const float* cent, const float* nrm, const float* verts,
    const float* area, const float* hsize, const float* hself_re,
    const float* hself_im, const float k, const float alpha_re,
    const float alpha_im, const float* signs, const float* offsets,
    const int first, const int last, const complex<float>* base, complex<float>* A,
    __half* half_re, __half* half_im, const float half_scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (j >= n) return;
    const float* x = cent + 3 * i;
    const float* nx = nrm + 3 * i;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v[9], c[3], ny[3];
    for (int image = first; image < last; ++image) {
        if (image == 0 && i == j) {
            acc[2] += hself_re[i];
            acc[3] += hself_im[i];
            continue;
        }
        load(verts, cent, nrm, j, signs + 3 * image, offsets + 3 * image, v, c, ny);
        element(x, nx, v, c, ny, area[j], hsize[j], k, 1, acc);
    }
    float re = -acc[0] - (alpha_re * acc[2] - alpha_im * acc[3]);
    float im = -acc[1] - (alpha_re * acc[3] + alpha_im * acc[2]);
    if (i == j && first == 0) re += 0.5f;
    complex<float> value(re, im);
    size_t index = (size_t)i * n + j;
    if (base != nullptr) value += base[index];
    if (half_re != nullptr) {  // scaled float16 copy only (real-time GMRES)
        half_re[index] = __float2half(value.real() * half_scale);
        half_im[index] = __float2half(value.imag() * half_scale);
    } else {
        A[index] = value;
    }
}

// value_p = sum_s w_s G(x_p, y_s), grad_p = its gradient in x_p (double precision)
extern "C" __global__ void monopoles(
    const int npts, const double* pts, const int nsrc, const double* src,
    const complex<double>* w, const double k, complex<double>* value, complex<double>* grad) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npts) return;
    complex<double> v(0.0, 0.0), g[3] = {complex<double>(0.0, 0.0), complex<double>(0.0, 0.0),
                                         complex<double>(0.0, 0.0)};
    for (int q = 0; q < nsrc; ++q) {
        double d[3] = {pts[3 * p] - src[3 * q], pts[3 * p + 1] - src[3 * q + 1],
                       pts[3 * p + 2] - src[3 * q + 2]};
        double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        double sn, cs;
        sincos(k * r, &sn, &cs);
        complex<double> green = complex<double>(cs, sn) * w[q] / (4.0 * PI * r);
        complex<double> radial = green * complex<double>(-1.0 / r, k) / r;  // dG/dr / r
        v += green;
        for (int a = 0; a < 3; ++a) g[a] += radial * d[a];
    }
    value[p] = v;
    for (int a = 0; a < 3; ++a) grad[3 * p + a] = g[a];
}

// D_pj = integral over element j of dG(x_p, y)/dn_y summed over all images
extern "C" __global__ void evaluation_matrix(
    const int npts, const float* pts, const int n, const float* cent,
    const float* nrm, const float* verts, const float* area, const float* hsize,
    const float k, const float* signs, const float* offsets, const int images,
    complex<float>* D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y;
    if (j >= n) return;
    const float* x = pts + 3 * p;
    float zero[3] = {0.0f, 0.0f, 0.0f};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v[9], c[3], ny[3];
    for (int image = 0; image < images; ++image) {
        load(verts, cent, nrm, j, signs + 3 * image, offsets + 3 * image, v, c, ny);
        element(x, zero, v, c, ny, area[j], hsize[j], k, 0, acc);
    }
    D[(size_t)p * n + j] = complex<float>(acc[0], acc[1]);
}

// out_p = sum_j u_j * D_pj, one thread block per point (threads stride over elements)
extern "C" __global__ void evaluate(
    const int npts, const float* pts, const int n, const float* cent,
    const float* nrm, const float* verts, const float* area, const float* hsize,
    const complex<float>* u, const float k, const float* signs, const float* offsets,
    const int images, complex<double>* out) {
    __shared__ float partial_re[128];
    __shared__ float partial_im[128];
    int p = blockIdx.x;
    if (p >= npts) return;
    const float* x = pts + 3 * p;
    float zero[3] = {0.0f, 0.0f, 0.0f};
    float sum_re = 0.0f, sum_im = 0.0f;
    float v[9], c[3], ny[3];
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int image = 0; image < images; ++image) {
            load(verts, cent, nrm, j, signs + 3 * image, offsets + 3 * image, v, c, ny);
            element(x, zero, v, c, ny, area[j], hsize[j], k, 0, acc);
        }
        float ur = u[j].real(), ui = u[j].imag();
        sum_re += acc[0] * ur - acc[1] * ui;
        sum_im += acc[0] * ui + acc[1] * ur;
    }
    partial_re[threadIdx.x] = sum_re;
    partial_im[threadIdx.x] = sum_im;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_re[threadIdx.x] += partial_re[threadIdx.x + stride];
            partial_im[threadIdx.x] += partial_im[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[p] = complex<double>(partial_re[0], partial_im[0]);
}
"""

_module = None


def module():
    global _module
    if _module is None:
        _module = cp.RawModule(code=_SOURCE, options=("--std=c++17",))
    return _module


def _images(images):
    signs, offsets = images
    return (cp.ascontiguousarray(cp.asarray(signs, dtype=cp.float32).ravel()),
            cp.ascontiguousarray(cp.asarray(offsets, dtype=cp.float32).ravel()), len(signs))


def assemble(geometry, k, alpha, hself, images, first=0, last=None, base=None, out=None,
             half_scale=None):
    """Matrix over image indices [first, last), added to ``base`` if given; image 0
    (the body itself) carries the 1/2 identity and the singular self terms. ``out``
    reuses existing output arrays. With ``half_scale`` the result is returned only as
    float16 (real, imaginary) parts multiplied by ``half_scale``."""
    n = geometry["n"]
    signs, offsets, count = _images(images)
    last = count if last is None else last
    if half_scale is None:
        A = cp.empty((n, n), dtype=cp.complex64) if out is None else out
        half = (cp.uint64(0), cp.uint64(0))
    else:
        A = None
        half = out if out is not None else (cp.empty((n, n), dtype=cp.float16),
                                            cp.empty((n, n), dtype=cp.float16))
    threads = 128
    module().get_function("assemble")(
        ((n + threads - 1) // threads, n), (threads,),
        (cp.int32(n), geometry["centroids"], geometry["normals"], geometry["vertices"],
         geometry["areas"], geometry["hsize"], cp.ascontiguousarray(hself.real.astype(cp.float32)),
         cp.ascontiguousarray(hself.imag.astype(cp.float32)), cp.float32(k), cp.float32(alpha.real),
         cp.float32(alpha.imag), signs, offsets, cp.int32(first), cp.int32(last),
         base if base is not None else cp.uint64(0), A if A is not None else cp.uint64(0),
         half[0], half[1], cp.float32(half_scale or 0.0)),
    )
    return A if half_scale is None else half


def monopoles(points, sources, weights, k):
    """Weighted monopoles: values (P,) and gradients (P, 3) at ``points`` (complex128)."""
    points = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float64))
    sources = cp.ascontiguousarray(cp.asarray(sources, dtype=cp.float64))
    weights = cp.ascontiguousarray(cp.asarray(weights, dtype=cp.complex128))
    value = cp.empty(len(points), dtype=cp.complex128)
    grad = cp.empty((len(points), 3), dtype=cp.complex128)
    threads = 128
    module().get_function("monopoles")(
        ((len(points) + threads - 1) // threads,), (threads,),
        (cp.int32(len(points)), points, cp.int32(len(sources)), sources, weights, cp.float64(k),
         value, grad),
    )
    return value, grad


def evaluation_matrix(geometry, points, k, images):
    """Dense (points x elements) double-layer matrix including all images."""
    points = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float32))
    n, count = geometry["n"], points.shape[0]
    signs, offsets, image_count = _images(images)
    D = cp.empty((count, n), dtype=cp.complex64)
    threads = 128
    module().get_function("evaluation_matrix")(
        ((n + threads - 1) // threads, count), (threads,),
        (cp.int32(count), points, cp.int32(n), geometry["centroids"], geometry["normals"],
         geometry["vertices"], geometry["areas"], geometry["hsize"], cp.float32(k),
         signs, offsets, cp.int32(image_count), D),
    )
    return D


def evaluate(geometry, points, u, k, images):
    points = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float32))
    count = points.shape[0]
    out = cp.empty(count, dtype=cp.complex128)
    signs, offsets, image_count = _images(images)
    module().get_function("evaluate")(
        (count,), (128,),
        (cp.int32(count), points, cp.int32(geometry["n"]), geometry["centroids"],
         geometry["normals"], geometry["vertices"], geometry["areas"], geometry["hsize"],
         cp.ascontiguousarray(u.astype(cp.complex64)), cp.float32(k), signs, offsets,
         cp.int32(image_count), out),
    )
    return out
