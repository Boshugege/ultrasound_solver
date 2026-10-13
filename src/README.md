# 代码使用说明


## 运行

使用默认配置：

```powershell
python src/main.py
```

指定配置文件：

```powershell
python src/main.py --config src/examples/config.yaml
```

运行远斜向目标点、`+x` 反射边界、传统方法对比示例：

```powershell
python src/main.py --config src/examples/phase_oblique_reflecting_x_16x16.yaml
```

## 可视化

列出结果文件中的可视化字段：

```powershell
python src/algorithm/visualizer.py outputs/phase_oblique_reflecting_x_16x16/result.npz --list-fields
```

查看优化后的振幅场：

```powershell
python src/algorithm/visualizer.py outputs/phase_oblique_reflecting_x_16x16/result.npz --field amplitude --percentile 95 --hide-source-layers 4
```

同屏对比传统几何相位和优化相位：

```powershell
python src/algorithm/visualizer.py outputs/phase_oblique_reflecting_x_16x16/result.npz --compare-methods --percentile 95 --hide-source-layers 4
```

红色等值面表示传统几何相位结果，蓝色等值面表示优化相位结果；两个滑条分别控制两种方法的等值面阈值。

## 批量算法对比

对固定场景依次运行几何相位、响应对齐、伴随梯度、GABS、SPSA、CMA-ES、L-SHADE、SAC 和 PPO：

```powershell
python src/algorithm/compare.py --config src/examples/phase_oblique_reflecting_x_16x16.yaml
```

默认会打开一个总 loss 面板；全部所选算法都会占用一个子图，算法数增加时自动增加行列，并将结果写入：

```text
outputs/algs/phase_oblique_reflecting_x_16x16/
  summary.csv
  summary.json
  loss_dashboard.png
  <algorithm>/config.yaml
  <algorithm>/result.npz
  <algorithm>/result_best_phases_rad.npy
  <algorithm>/result_final_phases_rad.npy
  <algorithm>/result_phase_export.json
  <algorithm>/result_loss.png
```

每次 `phase_optimization` 还会在 `result.npz` 同目录保存两个独立连续相位向量：

- `*_best_phases_rad.npy`：推荐使用的最优已评估相位。
- `*_final_phases_rad.npy`：优化停止时的最后一步相位。
- `*_phase_export.json`：相位单位、阵元数量、目标点、loss 和文件关联。

这些文件的相位单位均为 `rad`，仍是未校准、未量化的 solver 输出；发送前必须使用 AcousticField 的已标定通道顺序和相位修正，由 `src/hardware/` 的 Ultraino 服务量化并下发。

硬件接入使用本地 loopback 服务，而非让优化进程直接访问串口。服务默认 dry-run，仅监听 `127.0.0.1`；详细启动和发送方式见 [`hardware/README.md`](hardware/README.md)。

`--algorithms adjoint,gabs,spsa` 可只运行指定算法；`--no-loss-window` 用于无图形界面的批处理。SAC、PPO、CMA-ES 需要先安装 `src/algorithm/baselines/requirements.txt` 中的可选依赖。

默认每种迭代算法有相同的 `60 s` 墙钟时间上限；`--max-seconds 120` 可覆盖该比较预算。`iterations` 和 `max_evaluations` 只是各算法的安全上限，汇总表以实际用时、场评估数、VJP 数和 best loss 为准。

比较过程还会写入 TensorBoard 标量事件，不会自动打开或抢占浏览器。查看全部算法的交互式 loss 曲线：

```powershell
tensorboard --logdir outputs/algs/phase_oblique_reflecting_x_16x16/tensorboard
```

浏览已完成算法的三维振幅场：

```powershell
python src/algorithm/compare_visualizer.py outputs/algs/phase_oblique_reflecting_x_16x16
```

主视图的 `Algorithm index` 滑条切换算法，`Iso percentile` 滑条调节等值面阈值；原生下拉菜单可切换最终场、最佳已评估场、初始场和几何相位场。

## Config 参数

### 顶层

- `mode`: `phase_optimization` 根据 `algorithm` 优化阵元相位；`same_phase` 使用固定同相位前向求解；`sdf_inverse` 为未来 SDF 反演预留，当前会明确报未实现。
- `algorithm`: 相位优化算法，可选 `adjoint`、`geometric`、`response_alignment`、`gabs`、`spsa`、`lshade`、`sac`、`ppo`、`cmaes`。
- `algorithm_options`: 当前 `algorithm` 的专属参数映射；不属于该算法的参数会报错。
- `same_phase_rad`: `same_phase` 和 `sdf_inverse` 的所有阵元固定相位，单位 `rad`。

算法选择示例：

```yaml
mode: "phase_optimization"
algorithm: "spsa"
algorithm_options:
  learning_rate: 0.05
  perturbation: 0.1
  alpha: 0.602
  gamma: 0.101
```

完整示例见 `src/examples/algorithms/README.md`。

### `physics`

- `sound_speed`: 背景介质声速，单位 `m/s`。
- `medium_density`: 背景介质密度，单位 `kg/m^3`。

### `transducer_specs`

- `frequency`: 换能器频率，单位 `Hz`。
- `diameter`: 单个换能器有效直径，单位 `m`。
- `pitch`: 阵元中心间距，单位 `m`。
- `array_n`: 方形阵列单边阵元数量，总阵元数为 `array_n * array_n`。
- `geometry`: `ideal_grid` 或 16×16 的 `sonicsurface_reference`。
- `reference_anchor_m`: SonicSurface 参考坐标锚点 `[x,y]`，单位 `m`。
- `source_radial_order`: sinc Neumann 孔径的 Gauss 径向求积阶数。
- `source_angular_order`: sinc Neumann 孔径的方位角求积点数。

当前声源固定为刚性挡板上的归一化 `sinc_neumann` 有限孔径通量源。
已删除规定声压圆盘、`spl_db`、`spl_distance` 和声源模型选择分支。
公式、归一化、离散 RHS 与官方远场等价见
[`docs/SINC_NEUMANN_SOURCE.html`](../docs/SINC_NEUMANN_SOURCE.html)。

### `domain`

- `box_size`: 求解域尺寸 `[Lx, Ly, Lz]`，单位 `m`。
- `grid_size`: 网格点数 `[Nx, Ny, Nz]`。当前求解器要求三方向网格步长一致。
- `origin_m`: 求解域在物理坐标系中的原点 `[x,y,z]`，单位 `m`。

### `solver`

- `backend`: 求解后端，支持 `gpu_iterative`、`gpu_direct`、`cudss_hybrid_direct`。
- `direct_residual_tol`: 直接法相对残差检查阈值。
- `hybrid_num_threads`: cuDSS hybrid CPU 线程数。
- `hybrid_execution_mode`: cuDSS hybrid 模式，支持 `auto`、`hybrid_execute`、`hybrid_memory`。
- `hybrid_device_memory_limit_gb`: cuDSS hybrid 可使用的 GPU 显存上限，单位 `GB`。
- `hybrid_device_memory_reserve_gb`: 为其他 CUDA/Warp 用途预留的显存，单位 `GB`。
- `hybrid_register_cuda_memory`: 是否注册 CUDA 主存。
- `condensed_symmetry_tolerance`: 边界凝聚约化矩阵的复对称性检查阈值。
- `gmres_restart`: GMRES 每轮 Krylov 子空间大小。
- `gmres_maxiter`: GMRES 最大重启轮数。
- `gmres_rtol`: GMRES 相对残差阈值。
- `gmres_backward_error_tol`: 原系统后向误差阈值。
- `reuse_matrix`: 仅相位变化时是否复用矩阵装配结果。

### `training`

- `loss_type`: 损失函数，支持 `field_match`、`focal_pressure`、`focal_contrast`。
- `initial_phase`: 初始相位，支持 `geometric`、`response`、`current`、`zero`、`random`。
- `compare_geometric`: 是否额外计算并保存传统几何相位结果，用于红蓝对比可视化。
- `phase_basis_file`: 相位响应基缓存文件路径。
- `phase_basis_batch_size`: 构建响应基时每批处理的阵元数量。
- `voxel_chunk_size`: CPU 合成场和反向梯度计算的体素分块大小。
- `load_basis_to_gpu`: 是否将响应基常驻 GPU。
- `release_factor_after_basis`: 构建响应基后是否释放直接求解器因子。
- `target_field_file`: 外部目标振幅场文件；为空时根据 `targets` 自动生成高斯目标场。
- `target_peak_pressure`: 目标点峰值振幅，单位 `Pa`。
- `target_sigma`: 高斯目标场标准差，单位 `m`。
- `background_pressure`: 非目标区域期望振幅，单位 `Pa`。
- `target_weight`: 目标区域权重。
- `background_weight`: 背景/旁瓣抑制权重。
- `target_radius`: `focal_contrast` 中从旁瓣区域排除的目标半径，单位 `m`。
- `source_exclusion_layers`: 忽略底部源面附近的网格层数。
- `sidelobe_temperature`: `focal_contrast` 平滑最大旁瓣的温度参数，单位 `Pa`。
- `iterations`: 优化迭代次数。
- `max_evaluations`: 单次任务可用的总声场评估次数上限；GABS、SPSA、L-SHADE、RL、CMA-ES 和伴随法都使用此上限。
- `max_seconds`: 单次任务可用的墙钟时间上限，单位 `s`；`0` 表示不限制。`compare.py` 默认将其设为统一的 `60 s`。
- `seed`: 随机算法与 RL 的随机种子。
- `show_loss_curve`: 是否在相位优化时显示实时 loss 曲线窗口。
- `loss_curve_update_interval`: 每隔多少次真实场评估刷新曲线；横轴始终是场评估次数。
- `loss_curve_pause_seconds`: 每次 GUI 刷新的短暂停顿，单位 `s`。

### `algorithm_options`

- `adjoint`: 默认使用 `adam`；`optimizer` 支持 `lbfgsb`、`adam`、`adamw`、`lion`、`nonlinear_cg`。`learning_rate` 控制 Adam 步长，`convergence_patience` 与 `convergence_relative_tolerance` 控制 Adam/AdamW 的提前收敛判断；`gradient_check` 和 `gradient_check_step` 控制有限差分验证。
- `geometric`、`response_alignment`: 无专属参数。
- `gabs`: `phase_levels`，单个阵元每轮枚举的离散相位数。
- `spsa`: `learning_rate`、`perturbation`、`alpha`、`gamma`。
- `cmaes`: `sigma`、`population_size`；需要可选依赖。
- `lshade`: 成功历史自适应 Differential Evolution；`population_size` 为初始种群，`min_population_size` 为线性缩减后的下限，`memory_size` 为成功参数记忆长度，`p_best_rate` 为 current-to-pbest 候选比例。`convergence_patience` 与 `convergence_relative_tolerance` 控制连续代际相对改善不足时的收敛判断，默认连续 25 代相对改善不超过 `2e-4` 时停止。只访问标量损失，不使用解析梯度。
- `sac`、`ppo`: `episode_steps`、`evaluation_steps`、`action_scale`、`reward_scale`、`total_timesteps`、`checkpoint`、`run_mode` 等；需要可选依赖，完整参数见 `src/algorithm/baselines/requirements.txt` 和 `src/examples/algorithms/README.md`。

### `boundary_conditions`

- `-x`、`+x`、`-y`、`+y`、`-z`、`+z`: 六个外边界面，支持
  `open`、`reflecting`、`soft` 和 `{type: impedance, impedance: [R,X]}`。
- 生产阵列要求 `-z: reflecting`。阵元孔径在该刚性挡板上叠加非齐次
  Neumann 通量；孔径外保持零法向通量。`sinc_neumann` 是声源定义，
  不是额外的 YAML 边界类型。
- open 面可通过 `solver.pml` 配置内部 PML；源、目标与障碍物不得位于 PML 内。

### `targets`

- `point`: 目标点物理坐标 `[x, y, z]`，单位 `m`。可配置多个目标点。

### `obstacles`

- `type`: 障碍物类型，支持 `sphere`、`mesh`。
- `center` / `center_m`: 障碍物中心或变换后包围盒中心，单位 `m`。
- `radius`: 球体半径，单位 `m`。
- `file`: mesh 文件路径。
- `rotation_deg`: mesh 绕 `X/Y/Z` 轴旋转角，单位 `deg`。
- `scale`: mesh 原始单位到 `m` 的缩放比例。
- `smoothing_width_cells`: SDF 平滑界面半宽，单位为网格单元数。
- `material.density`: 障碍物密度，单位 `kg/m^3`。
- `material.sound_speed`: 障碍物声速，单位 `m/s`。

Mesh 障碍物示例：

```yaml
obstacles:
  - type: "mesh"
    file: "meshes/hand.stl"
    center_m: [0.05, 0.05, 0.045]
    rotation_deg: [0.0, 90.0, 0.0]
    scale: 0.001
    smoothing_width_cells: 1.2
    material:
      density: 1250.0
      sound_speed: 2200.0
```

`file` 使用相对路径时，以当前 YAML 文件所在目录为基准。mesh 需要是闭合、流形表面；求解器会自动采样为与 `domain.grid_size` 同分辨率的 SDF，且 SDF 内部为负值。详细说明见 `docs/MESH_TO_SDF.html`。

### `io`

- `output_file`: 结果 `.npz` 输出路径。
- `auto_visualize`: 求解后是否自动打开可视化。
- `colormap`: 振幅场可视化颜色表。
