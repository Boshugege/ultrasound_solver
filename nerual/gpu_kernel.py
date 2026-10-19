"""Fused CUDA boundary integration: one thread per receiver/panel pair."""
import numpy as np
import cupy as cp

_SOURCE = r'''
extern "C" __global__ void integrate(
    const float* points, const float* vertices, const float* normals,
    const float* areas, const float* diameters,
    const float* bary, const float* weights, const int nf, const int nn,
    const int m, const int n, const float k, const int image, float2* out) {
  int id=blockDim.x*blockIdx.x+threadIdx.x;
  if(id>=m*n) return;
  int row=id/n, col=id%n;
  const float* t=vertices+9*col;
  const float* normal=normals+3*col;
  float p[3]={points[3*row],points[3*row+1],points[3*row+2]};
  float dist=0, mirror_dist=0;
  for(int d=0;d<3;d++) {
    float center=(t[d]+t[3+d]+t[6+d])/3.f;
    float delta=p[d]-center;
    dist+=delta*delta;
    float md=p[d]-((d==2)?-center:center);
    mirror_dist+=md*md;
  }
  float threshold=1.8f*diameters[col];
  bool near=(dist<threshold*threshold || (image && mirror_dist<threshold*threshold));
  int start=near?nf:0, count=near?nn:nf;
  float re=0,im=0;
  for(int q=start;q<start+count;q++) {
    float y[3];
    for(int d=0;d<3;d++)
      y[d]=bary[3*q]*t[d]+bary[3*q+1]*t[3+d]+bary[3*q+2]*t[6+d];
    for(int mirror=0;mirror<=image;mirror++) {
      float r2=0,dot=0;
      for(int d=0;d<3;d++) {
        float sign=(mirror && d==2)?-1.f:1.f;
        float delta=p[d]-sign*y[d];
        r2+=delta*delta;
        dot+=delta*normal[d]*sign;
      }
      if(fabsf(dot)<1.e-8f) continue;
      float r=sqrtf(r2), kr=k*r, sn,cs;
      sincosf(kr,&sn,&cs);
      float factor=dot*weights[q]*areas[col]/(12.566370614359f*r*r*r);
      re+=factor*(cs+kr*sn);
      im+=factor*(sn-kr*cs);
    }
  }
  out[id]=make_float2(re,im);
}
'''
_kernel = cp.RawKernel(_SOURCE, "integrate")


def integrate(points, surface, k, bary_far, weights_far, bary_near, weights_near, image=True):
    points = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float32))
    triangles = cp.asarray(surface.triangles, dtype=cp.float32)
    normals = cp.asarray(surface.normals, dtype=cp.float32)
    areas = cp.asarray(surface.areas, dtype=cp.float32)
    diameters = cp.asarray(surface.diameters, dtype=cp.float32)
    bary = cp.asarray(np.vstack([bary_far,bary_near]), dtype=cp.float32)
    weights = cp.asarray(np.r_[weights_far,weights_near], dtype=cp.float32)
    m,n = len(points),len(triangles)
    output = cp.empty((m,n),dtype=cp.complex64)
    _kernel(((m*n+127)//128,), (128,),
            (points,triangles,normals,areas,diameters,bary,weights,
             np.int32(len(weights_far)),np.int32(len(weights_near)),
             np.int32(m),np.int32(n),np.float32(k),np.int32(image),output))
    return output
