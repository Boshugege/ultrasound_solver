# 固定手模型相位代理网络数据获取流程

## 1. 目标

训练一个固定手模型的神经代理模型：

```text
输入：手根位姿 + 关节姿态 + 目标点
输出：256 通道复响应或相位
```

部署时使用：

```text
phase_j = -arg(response_j)
```

本流程只解决目标点聚焦相位，不做手部几何反推。

核心约束：

- 手模型固定，骨长、拓扑、蒙皮权重、关节定义不变。
- 换能器阵列、频率、空气参数和硬件通道顺序不变。
- 训练标签由 BEM 教师生成，不使用 PDE 或直接距离法作为标签。
- 数据集保存 256 通道复响应，不只保存相位。
- 训练集和评测集必须按姿态、轨迹和手势族拆分，不能按单个目标点随机拆分。

## 2. 前置产物

当前静态手模型：

```text
obstacle_displacement_study/models/hand_watertight.vtk
```

需要先建立固定手 rig：

```text
outputs/hand_rig_v1/
  template_vertices.npy
  faces.npy
  joint_rest_positions.npy
  joint_parents.npy
  joint_limits.yaml
  skinning_weights.npy
  rig_manifest.json
```

`rig_manifest.json` 必须记录：

```text
hand_mesh_path
vertex_count
face_count
joint_names
joint_dof_names
joint_limit_source
skinning_method
coordinate_frame
created_git_commit
sha256(template_vertices)
sha256(faces)
sha256(skinning_weights)
```

在 rig 未冻结前，不生成正式训练集。

## 3. 数据集目录

建议第一版目录：

```text
outputs/hand_phase_surrogate_v1/
  manifest.json
  generation_config.yaml
  rig_manifest.json
  splits.json
  poses/
    train_poses_000.npz
    val_poses_000.npz
    test_id_poses_000.npz
    test_ood_poses_000.npz
  responses/
    train_000.npz
    val_000.npz
    test_id_000.npz
    test_ood_000.npz
  audit/
    rejected_poses.jsonl
    rejected_targets.jsonl
    convergence_summary.json
    teacher_refinement_checks.json
    dataset_statistics.json
```

`manifest.json` 必须记录：

```text
dataset_version
rig_version
teacher_solver
teacher_config_hash
transducer_geometry_hash
channel_order
frequency_hz
sound_speed_m_s
medium_density
git_commit
cuda_version
cupy_version
created_at
random_seed
```

## 4. 训练集数据获取流程

### 4.1 冻结训练分布

先写入 `generation_config.yaml`，不要在生成中途手动改参数。

训练分布包括：

```text
root_translation_range_m
root_rotation_range_deg
joint_angle_distribution
gesture_template_weights
target_sampling_distribution
reflector_or_scene_config
mesh_quality_thresholds
bem_teacher_config
```

第一版建议：

```yaml
root_pose:
  center_m: [0.08, 0.082, 0.1165]
  rotation_deg: [0.0, 90.0, 90.0]
  translation_jitter_m: [0.020, 0.020, 0.020]
  yaw_jitter_deg: 30.0
  pitch_jitter_deg: 20.0
  roll_jitter_deg: 20.0

targets_per_pose: 32

target_mix:
  palm_near: 12
  fingertip_near: 10
  occlusion_hard: 6
  workspace_uniform: 4
```

### 4.2 采样手姿态

每个 `pose_id` 生成一次完整手姿态：

```text
root_translation_m     [3]
root_rotation_6d       [6]
joint_angles_rad       [J]
joint_positions_world_m[21, 3]
gesture_family         int
trajectory_id          int or -1
```

姿态来源按混合分布采样：

| 来源 | 建议比例 | 作用 |
|---|---:|---|
| Sobol 关节采样 | 40% | 覆盖关节空间 |
| 手势模板扰动 | 25% | 保证自然手势 |
| 协同/PCA 姿态 | 20% | 减少不自然组合 |
| 连续轨迹插值 | 10% | 覆盖动态路径 |
| 边界压力测试 | 5% | 覆盖接近关节限位样本 |

采样后立刻做几何检查：

- 关节角在限位内。
- 生成网格闭合，面元顺序不变。
- 无退化三角形。
- 无明显自相交。
- 手不穿过阵列底板。
- 手和已启用反射面保持安全距离。

失败样本写入：

```text
audit/rejected_poses.jsonl
```

每行包含：

```json
{"pose_id": 17, "reason": "self_intersection", "source": "sobol"}
```

### 4.3 为每个姿态采样目标点

对通过几何检查的每个手姿态，采样多个目标点。

每个目标点保存：

```text
sample_id
pose_id
target_world_m       [3]
target_hand_local_m  [3]
target_category
nearest_surface_mm
visibility_hint
```

目标点类别：

| 类别 | 采样方式 | 目的 |
---|---|---|
| 掌面附近 | 掌面外法向偏移 5-20 mm | 常见交互区域 |
| 指腹/指尖附近 | 指尖、指腹外侧偏移 3-15 mm | 最需要动态跟随 |
| 遮挡困难点 | 指缝、弯曲手指声影区 | 提高 BEM 优势场景覆盖 |
| 工作空间均匀点 | 阵列上方合法空间采样 | 防止只会贴手目标 |

目标点检查：

- 不在手网格内部。
- 距离手表面大于教师积分安全下限。
- 不落在反射面或底板上。
- 不低于阵列安全高度。
- 目标点处理论上限 `sum(abs(h_j))` 不过低。

失败样本写入：

```text
audit/rejected_targets.jsonl
```

### 4.4 生成 BEM 教师标签

对每个手姿态：

1. 根据 `root pose + joint angles` 生成形变手网格。
2. 将手网格作为刚性障碍物加入 BEM 场景。
3. 如果反射面配置固定，也加入同一个场景。
4. 完整装配一次 BEM 矩阵。
5. 对该姿态的多个目标点调用 `target_response(points)`。
6. 保存每个目标点的 256 通道复响应。
7. 保存残差、耗时和教师配置元数据。
8. 释放该姿态的 BEM 对象，进入下一个姿态。

标签保存：

```text
response_real [N, 256] float32
response_imag [N, 256] float32
```

不要只保存：

```text
phase [N, 256]
```

原因：

- 相位有 `2*pi` 跳变。
- 复响应可以重新生成连续相位、32 档相位和多目标叠加。
- 评测时可以直接计算聚焦效率。

每个样本同时保存：

```text
gmres_residual
solve_ms
assembly_ms
focus_upper_bound = sum(abs(response))
teacher_status
```

### 4.5 教师精度设置

不要用实时路径作为最终标签。

Pilot 阶段可用：

```text
surface resolution: lambda/4
GMRES rtol: 1e-5
matrix precision: complex64
```

正式训练集建议：

```text
surface resolution: lambda/5
GMRES rtol: 1e-6 or validated 1e-5
matrix precision: complex64
quadrature: fine
```

当前 31 ms 实时 BEM 路径的 `rtol=1e-2`、FP16 矩阵存储适合在线控制，不适合作为最终教师真值。

### 4.6 分片写入

响应数据按分片写入，避免单文件过大：

```text
responses/train_000.npz
responses/train_001.npz
...
```

每个分片建议 2,000-10,000 个样本。

分片字段：

```text
sample_id              int64   [N]
pose_id                int64   [N]
target_world_m         float32 [N,3]
target_hand_local_m    float32 [N,3]
target_category        uint8   [N]
response_real          float32 [N,256]
response_imag          float32 [N,256]
gmres_residual         float32 [N]
solve_ms               float32 [N]
assembly_ms            float32 [N]
focus_upper_bound      float32 [N]
teacher_status         uint8   [N]
```

姿态字段独立保存，避免重复：

```text
pose_id                  int64   [P]
root_translation_m       float32 [P,3]
root_rotation_6d         float32 [P,6]
joint_angles_rad         float32 [P,J]
joint_positions_world_m  float32 [P,21,3]
gesture_family           uint8   [P]
trajectory_id            int64   [P]
split                    uint8   [P]
```

### 4.7 训练集规模

分三阶段生成。

| 阶段 | 姿态数 | 每姿态目标 | 样本数 | 目标 |
|---|---:|---:|---:|---|
| Pilot | 200 | 16 | 3,200 | 验证管线、耗时和拒绝率 |
| MVP | 5,000 | 32 | 160,000 | 训练第一个可用模型 |
| Full | 20,000 | 32 | 640,000 | 正式模型和消融 |

Pilot 完成前，不承诺 Full 生成时间。必须先记录：

- 每姿态装配耗时。
- 每目标响应耗时。
- GPU 显存峰值。
- 姿态拒绝率。
- 目标拒绝率。
- BEM 不收敛率。

## 5. 评测集数据获取流程

评测集不能从训练集随机抽目标点得到。评测集要在生成前定义并冻结。

建议拆成四类。

### 5.1 Validation

用途：训练过程中的模型选择和早停。

来源：

- 和训练集同分布。
- 姿态 ID 与训练集完全不同。
- 连续轨迹不能跨 train/validation。

规模建议：

```text
2,000 poses x 32 targets = 64,000 samples
```

可以用同一教师精度生成。

### 5.2 Test-ID

用途：报告训练分布内真实性能。

要求：

- 采样分布和训练集一致。
- 姿态、轨迹、手势模板实例与训练集完全隔离。
- 不能参与模型选择。
- 不能根据 Test-ID 结果修改网络或采样分布。

规模建议：

```text
2,000 poses x 32 targets = 64,000 samples
```

Test-ID 必须用最终教师配置生成，不能用 Pilot 教师。

### 5.3 Test-OOD

用途：评估训练分布外但仍可能部署遇到的情况。

OOD 不等于非法输入。非法输入应回退 BEM 或关闭输出。

建议包含：

| OOD 类型 | 示例 |
|---|---|
| 未见手势族 | 训练集中不出现的完整手势模板 |
| 关节边界 | 接近但不超过关节限位 |
| 根位姿边界 | 更靠近阵列边缘或更大旋转 |
| 困难目标点 | 指缝、手背声影区、近遮挡点 |
| 反射面敏感区 | 靠近固定反射面或边界 |

规模建议：

```text
2,000 poses x 32 targets = 64,000 samples
```

OOD 样本要分组保存 `ood_type`，不能只给总分。

### 5.4 Stress / Safety

用途：部署门控，不用于训练。

包含：

- 跟踪器可能给出的极端关节角。
- 接近自碰撞的手势。
- 目标点过近手表面。
- 目标点接近反射面、底板或阵列边缘。
- 低响应上限区域。

这些样本的目标不是要求网络都预测准确，而是验证：

```text
模型置信度/输入检查能否触发回退
```

评估时分成两类：

- 合法但困难：报告聚焦效率。
- 非法或不可信：报告回退率和误放行率。

## 6. 评测集冻结规则

生成评测集前先冻结：

```text
splits.json
```

内容包括：

```text
train_pose_ids
validation_pose_ids
test_id_pose_ids
test_ood_pose_ids
stress_pose_ids
trajectory_to_split
gesture_family_to_split
random_seed
```

硬规则：

- 同一个 `pose_id` 的所有目标只能在一个 split。
- 同一条连续轨迹的所有帧只能在一个 split。
- 近邻姿态聚类后，同一簇不能跨 train/test。
- OOD 保留的手势族不能泄漏进训练集。
- Test-ID 和 Test-OOD 生成后不再重采样。

如果发现评测集有错误，必须升级数据集版本，例如：

```text
hand_phase_surrogate_v2
```

不要静默覆盖 v1。

## 7. 评测指标

主指标是教师复响应下的聚焦效率：

```text
eta = abs(sum_j h_j * exp(i * predicted_phase_j))) / sum_j abs(h_j)
```

必须报告：

| 指标 | 分组 |
|---|---|
| 连续相位聚焦效率 mean / P50 / P5 / worst | split、手势族、目标类别 |
| 32 档量化后聚焦效率 mean / P50 / P5 / worst | split、手势族、目标类别 |
| 复响应相位误差，按 `abs(h_j)` 加权 | split、目标类别 |
| 目标点峰值相对 BEM | split、困难区域 |
| 推理耗时 P50 / P95 | GPU、CPU |
| 回退率 | OOD、Stress |
| 误放行率 | Stress |

第一版建议验收门槛：

```text
Test-ID continuous phase P5 >= 99.5%
Test-ID 32-step phase P5 >= 99.0%
Test-OOD 单独报告，不用 ID 指标替代
Stress 非法输入误放行率接近 0
```

## 8. 数据生成执行顺序

### 阶段 A：rig 和姿态生成验证

1. 冻结 fixed hand rig。
2. 生成 1,000 个候选姿态。
3. 做几何检查和可视化抽查。
4. 统计拒绝原因。
5. 调整采样分布，直到无效姿态比例可控。

产物：

```text
outputs/hand_phase_surrogate_v1/audit/pose_sampling_report.json
```

### 阶段 B：Pilot 标签生成

1. 固定 200 个有效姿态。
2. 每个姿态采 16 个目标。
3. 用 Pilot 教师生成复响应。
4. 抽样做更严格教师复核。
5. 统计耗时、显存、不收敛率和标签质量。

产物：

```text
outputs/hand_phase_surrogate_v1/audit/pilot_report.json
```

只有 Pilot 通过后才进入 MVP。

### 阶段 C：冻结 schema 和 split

1. 冻结 `generation_config.yaml`。
2. 冻结 `manifest.json`。
3. 冻结 `splits.json`。
4. 记录 Git commit 和教师配置 hash。

之后不得继续修改 split。

### 阶段 D：MVP 数据生成

1. 生成 5,000 个训练姿态。
2. 生成 validation、test-id、test-ood 姿态。
3. 每姿态 32 个目标。
4. 生成复响应标签。
5. 训练第一个代理模型。
6. 用 validation 调参。
7. 只在模型冻结后跑 test。

### 阶段 E：Full 数据生成

1. 扩展到 20,000 个训练姿态。
2. 保留独立测试集不变或升级版本。
3. 生成正式报告。
4. 输出部署模型和回退策略。

## 9. 不应做的事

- 不要把同一姿态下不同目标随机分到 train/test。
- 不要只保存相位，不保存复响应。
- 不要用实时 FP16 BEM 作为最终教师。
- 不要根据 Test-ID 或 Test-OOD 结果反复改训练分布。
- 不要把非法 Stress 样本计入普通均值来掩盖回退问题。
- 不要在 rig 未冻结时生成正式数据。

## 10. 最小实现接口建议

后续代码可以拆成这些脚本：

```text
scripts/hand_dataset/
  sample_poses.py
  validate_pose_meshes.py
  sample_targets.py
  generate_bem_labels.py
  split_dataset.py
  audit_dataset.py
  evaluate_surrogate.py
```

最小命令形态：

```powershell
python scripts\hand_dataset\sample_poses.py --config configs\hand_dataset_v1.yaml
python scripts\hand_dataset\sample_targets.py --dataset outputs\hand_phase_surrogate_v1
python scripts\hand_dataset\generate_bem_labels.py --dataset outputs\hand_phase_surrogate_v1 --split train
python scripts\hand_dataset\audit_dataset.py --dataset outputs\hand_phase_surrogate_v1
```

`generate_bem_labels.py` 内部应调用当前 BEM 路径：

```text
SimulationConfig.from_yaml(...)
BemSceneSolver(config).target_response(targets)
```

每个姿态只装配一次 BEM 场景，然后批量求该姿态下的所有目标点响应。
