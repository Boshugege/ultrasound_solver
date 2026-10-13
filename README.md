# Ultrasound Solver

三维变参数 Helmholtz 超声场求解与相位优化项目。

项目当前主线是在固定障碍物、介质分布、边界条件和换能器幅值的场景下，求解完整三维复声压场，并优化换能器阵列相位，使指定目标点或目标区域形成更强振幅聚焦，同时抑制非目标区域旁瓣。

## 核心能力

- 三维 Helmholtz 方程 7 点 stencil 离散。
- 支持变密度、变声速介质和基于 SDF 的障碍物建模。
- 使用刚性挡板上的归一化 `sinc_neumann` 有限孔径通量源；支持局部 open、
  reflecting、soft、impedance 边界及 PML。
- 支持多种求解后端：`gpu_iterative`、`gpu_direct`、`cudss_hybrid_direct`。
- 支持边界凝聚直接法，降低大规模直接求解的显存压力。
- 支持固定场景的相位响应基：

```text
u(phi) = G exp(i phi)
```

- 支持相位算法选择：伴随梯度、几何相位、响应对齐、GABS、SPSA，以及可选 SAC、PPO、CMA-ES。
- 支持 `field_match`、`focal_pressure`、`focal_contrast` 等目标，并可在同一 PyVista 窗口中红蓝等值面对照几何相位与优化结果。
- 支持目标振幅场自动生成、有限差分梯度检查和交互式三维可视化。

## 仓库结构

```text
src/
  README.md                  代码运行说明和 config 参数定义
  main.py                    顶层入口：编排算法与后续硬件部署
  algorithm/                 数值声学建模、相位优化与可视化
    config.py                YAML 配置解析
    physics/                 换能器、SDF、矩阵和 RHS 装配
    solvers/                 Helmholtz 线性系统求解后端
    training/                响应基、目标场、loss 和梯度公式
    baselines/               可选相位算法及统一评估接口
    compare.py               批量算法对比入口
  hardware/                  阵列协议、映射、校准与安全上传适配器
  examples/                  默认配置和可复现实验配置

docs/
  main.html                  文档总入口
  *.html                     物理推导、离散化、边界凝聚、梯度和架构文档

obstacle_displacement_study/ 障碍物轻移实验脚本与结果整理
algorithm_comparison/        可上传的算法对比配置、结果与曲线
gradient_ablation/           可上传的解析梯度更新器消融实验
outputs/                     响应基、优化结果和可视化产物，不提交
requirements.txt             Python 依赖列表
```

## 快速使用

详细运行方式和 config 参数定义见：

- [`src/README.md`](src/README.md)

运行默认配置：

```powershell
python src/main.py
```

运行远斜向目标点、`+x` 反射边界、传统方法对比示例：

```powershell
python src/main.py --config src/examples/phase_oblique_reflecting_x_16x16.yaml
```

同屏查看传统几何相位与优化相位结果：

```powershell
python src/algorithm/visualizer.py outputs/phase_oblique_reflecting_x_16x16/result.npz --compare-methods --percentile 95 --hide-source-layers 4
```

## 文档

建议从文档总入口开始阅读：

- [`docs/main.html`](docs/main.html)

主要专题文档：

- [`docs/SINC_NEUMANN_SOURCE.html`](docs/SINC_NEUMANN_SOURCE.html)：当前唯一声源模型、
  Neumann 孔径公式、官方 sinc 远场等价与现实适用边界。
- [`docs/ACOUSTIC_PHYSICS_DERIVATION.html`](docs/ACOUSTIC_PHYSICS_DERIVATION.html)：从流体力学到声学 Helmholtz 方程。
- [`docs/HELMHOLTZ_DISCRETIZATION.html`](docs/HELMHOLTZ_DISCRETIZATION.html)：离散化、矩阵装配和边界处理。
- [`docs/MESH_TO_SDF.html`](docs/MESH_TO_SDF.html)：mesh 障碍物配置、几何变换、SDF 采样和材料插值。
- [`docs/BOUNDARY_CONDENSATION.html`](docs/BOUNDARY_CONDENSATION.html)：边界凝聚直接法。
- [`docs/PHASE_ONLY_TRAINING.html`](docs/PHASE_ONLY_TRAINING.html)：固定场景下的相位响应基训练。
- [`docs/GRADIENT_DERIVATION.html`](docs/GRADIENT_DERIVATION.html)：振幅 loss 导数、相位 VJP、PDE 伴随梯度和有限差分验证。
- [`docs/SOLVER_ARCHITECTURE.html`](docs/SOLVER_ARCHITECTURE.html)：求解器架构和模块边界。
- [`algorithm_comparison/README.md`](algorithm_comparison/README.md)：已完成的 9 种算法对比、质量门槛统计和复现实验入口。
- [`gradient_ablation/README.md`](gradient_ablation/README.md)：已完成的解析梯度更新器消融、质量门槛统计和复现实验入口。
- [`src/hardware/README.md`](src/hardware/README.md)：Ultraino/AcousticField 标定结果与本地 S/C 硬件服务的接入方式。

## 典型工作流

1. 在 `src/examples/` 中选择或复制一个 YAML 配置。
2. 设置求解域、阵列规模、障碍物、边界条件和目标点。
3. 使用 `python src/main.py --config ...` 运行前向求解或 phase-only 优化。
4. 使用 `src/algorithm/visualizer.py` 查看 `.npz` 结果文件；同目录自动生成推荐部署的 `*_best_phases_rad.npy` 与相位 manifest。
5. 将可复现实验配置保留在 `src/examples/`，将运行产物保留在 `outputs/`。

## 产物约定

- `outputs/`、`.npy`、`.npz`、响应基缓存和截图属于运行产物，默认不提交。
- `algorithm_comparison/` 是对外发布的例外，保留本次算法对比的配置、结果和曲线，可直接上传。
- `gradient_ablation/` 同样是对外发布的例外，保留优化器消融的配置、结果和曲线；大型响应基缓存不提交。
- 可复现实验应提交 YAML 配置，而不是提交大型结果文件。
- `src/README.md` 只维护代码使用方式和 config 参数定义。
- 根目录 `README.md` 只维护项目概览、结构和文档入口。
