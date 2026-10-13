# sinc Neumann 有限孔径声源

日期：2026-10-12。

严格的半空间 Green 函数、孔径傅里叶变换和官方远场等价证明见
[sinc Neumann 有限孔径与官方模型的等价推导](SINC_NEUMANN_SOURCE.html)。

本文档定义当前生产 PDE 使用的唯一阵元声源模型。旧的规定声压圆盘
（`pressure_disc`、换能器 Dirichlet 行和 SPL→表面声压换算）已经删除，
不得再作为当前实现说明。

## 1. 术语和边界组成

`sinc_neumann` 不是六个外边界中的一种 YAML 类型，而是位于底面刚性挡板上的
**非齐次 Neumann 有限孔径声源**：

- 阵列位于 `-z` 面；
- 阵元孔径之外满足刚壁条件 `(1/rho) dp/dn = 0`；
- 每个阵元孔径内叠加归一化法向通量；
- `±x`、`±y`、`+z` 为 open，并在生产配置中设置 PML。

因此底面可统一理解为“带局部非零通量的反射挡板”，而不是
“Dirichlet 圆盘加其余反射面”。

项目采用时间约定：

```text
Re(p exp(-i omega t))
```

域内方程为：

```text
div((1/rho) grad p) + omega^2/(rho c^2) p = 0
```

## 2. 孔径通量分布

设阵元编号为 `j=1,...,M`，第 `j` 个阵元中心为 `c_j`。对于孔径面上的点 `s`，
定义局部径向距离：

```text
varrho_j(s) = |s_parallel-c_j|
```

阵元直径为 `d`、半径为 `a=d/2`。所有阵元共用同一个径向孔径函数：

```text
f_a(varrho) = 1 / [2 pi a sqrt(a^2-varrho^2)],  0 <= varrho < a
```

第 `j` 个阵元的外向法向通量为：

```text
(1/rho) dp/dn =
    (2 pi/rho_ref) A_j exp(i phi_j) f_a(varrho_j(s)),
    s in Gamma_j
```

其中 `A_j` 和 `phi_j` 分别是阵元驱动幅值与相位，`rho_ref` 是装配源项时固定的
参考密度。底面外法向为 `-z`，所以离散 RHS 中会出现相应的符号；这只是法向约定，
不改变上述物理定义。

该分布在孔径边缘具有可积奇性，但总质量严格为 1：

```text
integral_Gamma_j f_a(varrho_j(s)) dS = 1
```

因此每个阵元的归一化源强不依赖网格覆盖点数量。

## 3. 为什么远场是官方 sinc

孔径分布的二维傅里叶变换为：

```text
F_a(q) = sin(a |q|)/(a |q|)
       = sinc(a |q|/pi)
```

其中 `sinc(x)=sin(pi x)/(pi x)`。观察方向与阵元法线夹角为 `theta` 时，
切向波数 `|q|=k sin(theta)`，所以：

```text
D(theta) = sin(a k sin(theta))/(a k sin(theta))
         = sinc(a k sin(theta)/pi)
```

有限孔径的精确自由场积分为：

```text
p_j(x) =
    A_j exp(i phi_j)
    integral_Gamma_j f_a(s) exp(i k |x-s|)/|x-s| dS
```

在 `R >> a` 的远场条件下：

```text
p_j(x) ~
    A_j exp(i phi_j)
    D(theta) exp(i k R)/R
```

右式正是当前官方参考传播公式。因此官方公式是该有限孔径积分的远场渐近式；
PDE 不是把 sinc 函数直接写到边界，而是通过具有该傅里叶变换的 Neumann
孔径通量，在近场、反射和变材料场景中继续求解完整波动方程。

## 4. 离散实现

实现文件：

- `src/algorithm/physics/sinc_aperture.py`
- `src/algorithm/physics/warp_utils.py`
- `src/algorithm/solvers/helmholtz_solver.py`
- `src/algorithm/solvers/homogeneous_surface.py`

`source_mass_matrix()` 使用 Gauss 径向求积和均匀方位角求积。变量替换
`mu=cos(theta)` 消去边缘可积奇性，求积点不会直接落在 `r=a`。
每个求积点再双线性沉积到四个源面网格节点，从而保持：

- 每列总质量为 1；
- 非网格阵元的质心不漂移；
- 网格加密时孔径分布收敛；
- 各阵元幅值和相位保持线性叠加。

将归一化离散通量列记为 `S`，阵元幅值记为 `A_j`，定义包含固定幅值的源矩阵：

```text
B = S diag(A)
w_j(phi) = exp(i phi_j)
```

完整系统为：

```text
A p(phi) = B w(phi)
```

左侧系统矩阵 `A` 包含介质、外边界和 PML；源矩阵 `B` 是底面有限孔径通量对应的
保守有限体积 RHS。当前所有阵元幅值均为 1，因此数值上 `B=S`。
相位变化只更新 `w(phi)` 和 RHS，不改变固定场景的矩阵 `A`。

源项不能与侧边界、PML 或 pressure-release (`soft`) 面重叠。生产求解器会在装配时
检查这些冲突。均匀自由场的表面凝聚与完整三维直接求解使用同一个 `B`，
不存在第二套解析声源后端。

## 5. 与现实硬件的关系

线性声学中，表面法向速度与声压法向导数满足：

```text
v_n = -(1/(i omega rho)) dp/dn
```

因此 Neumann 通量是有限面积辐射表面的自然等效输入。它比强制固定表面声压更符合
“阵元由电信号驱动并向刚性 PCB 上方辐射”的机制；孔径之外的 PCB 则近似刚性挡板。

但是 `sinc_neumann` 仍是等效声学模型，不是压电换能器的完整数字孪生。它未显式表示：

- 振膜机械模态与频率响应；
- 阵元间机械/电气互耦；
- 黏热损耗和结构振动；
- 个体幅相误差与绝对声压标定。

所以“现实对齐”应理解为：源变量、有限孔径、挡板和方向图具有正确的物理结构，
并与官方可复现模型达到工程一致；绝对 Pa 与个体硬件仍需麦克风扫描和逐阵元标定。

## 6. 数值对齐证据

在 40 kHz、声速 346 m/s、直径 9 mm、1/3 mm 网格下：

- 中心与偏轴目标的 PDE/官方最优振幅差约 `+0.53% / +0.48%`；
- 官方相位输入 PDE 后达到 PDE 单点上限的 `99.928% / 99.912%`；
- PDE 相位输入官方模型后达到官方上限的 `99.928% / 99.912%`；
- 扩展目标的官方模型振幅效率为 `99.879%–99.921%`。

这些结果验证了官方远场模型与当前有限孔径 PDE 的工程一致性，但不替代硬件绝对声压验收。
完整数据见 [PDE_PHASE_ALIGNMENT.md](PDE_PHASE_ALIGNMENT.md)。

## 7. 配置

声源模型是当前求解器的固定物理定义，不再提供旧 `source_model` 选择项。阵元相关配置为：

```yaml
transducer_specs:
  frequency: 40000.0
  diameter: 0.009
  pitch: 0.0105
  array_n: 16
  geometry: sonicsurface_reference
  reference_anchor_m: [0.08, 0.08]
  source_radial_order: 64
  source_angular_order: 256

boundary_conditions:
  "-x": open
  "+x": open
  "-y": open
  "+y": open
  "-z": reflecting
  "+z": open
```

`source_radial_order` 与 `source_angular_order` 控制孔径求积精度，不改变物理孔径尺寸。
