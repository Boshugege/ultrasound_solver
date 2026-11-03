# 手部姿态参数数据集选择

## 结论

当前过程性手势只能用于验证 BEM/可视化管线，不应继续作为训练集或演示手势来源。

后续应切换到 MANO 参数数据：

```text
MANO pose / shape / trans -> 手网格姿态 -> BEM 场景 -> 256 通道响应
```

第一优先级建议使用：

```text
InterHand2.6M annotation-only MANO fits
```

原因：

- 官方数据页明确提供 `InterHand2.6M_$SPLIT_MANO_NeuralAnnot.json`；
- 每帧含 `pose`、`shape`、`trans`；
- `pose` 是 48 维 axis-angle MANO pose；
- `shape` 是 10 维 MANO shape；
- `trans` 是米单位 3D 平移；
- 可只下载 annotation，不需要 80GB/365GB 图像。

## 数据集候选

| 数据集 | 是否有 MANO 参数 | 优点 | 问题 | 推荐级别 |
|---|---|---|---|---|
| MANO model | 是模型，不是姿态集 | 标准手模型，低维参数化，适合生成网格 | 需要注册和同意 license；本身不提供大量动作样本 | 必需 |
| InterHand2.6M | 是 | 官方文档明确提供 48D pose、10D shape、3D trans；有单手和双手；annotation 可单独下载 | 数据量很大，需要筛选单手/右手/高质量帧 | 第一优先级 |
| FreiHAND | 是 | 有真实图像、3D pose 和 shape annotation；常用 benchmark | 手势多样性不如交互视频；可能更偏单手静态 | 第二优先级 |
| HanCo | 有 predicted MANO fit | 多视角、时序数据；102,480 time steps 有 MANO fit，其中 9,055 人工检查 | 部分 MANO fit 非人工验证，需过滤质量 | 第二优先级 |
| OakInk | 是 | 抓握/手物交互丰富，含意图和物体 | 手势受物体交互影响，数据接入稍复杂 | 后续扩展 |
| DexYCB | 使用 MANO | 抓握 YCB 物体，真实交互序列 | 数据大，主要是 grasp/object 场景；license 非商用 | 后续扩展 |
| HOT3D | 是 | 新的高质量 egocentric 数据，提供 MANO/UmeTrack | 数据体系复杂，主要面向手物 tracking | 后续扩展 |

## 推荐路线

### 阶段 1：改用 MANO 作为固定手模型

最稳的方式是直接把 MANO 网格作为我们的固定手模型：

```text
固定 shape beta = beta0
输入 pose theta + root transform + target point
输出 BEM response / phase
```

优点：

- 数据集 pose 参数可直接驱动网格；
- 不需要把 MANO 姿态重新绑定到 `hand_watertight.vtk`；
- 关节定义、父子层级、蒙皮权重都有标准实现；
- 后续训练数据与公开姿态数据完全一致。

缺点：

- 当前 `hand_watertight.vtk` 的 BEM 收敛证据不能完全复用；
- 需要对 MANO 网格重新做 BEM 网格质量和尺度验证。

### 阶段 2：如果必须沿用当前 `hand_watertight.vtk`

需要做 retarget：

```text
MANO joints/mesh -> 当前 hand_watertight.vtk rig
```

必须补充：

- 当前手网格的 21 关节骨架；
- 蒙皮权重；
- MANO joint 到当前 rig joint 的映射；
- 尺度和坐标系校准；
- 至少 20 个 MANO 姿态的可视化验证。

这条路径工作量更大，但能保留当前手网格。

## InterHand2.6M 接入流程

只下载 annotations：

```text
annotations/
  InterHand2.6M_train_MANO_NeuralAnnot.json
  InterHand2.6M_val_MANO_NeuralAnnot.json
  InterHand2.6M_test_MANO_NeuralAnnot.json
  InterHand2.6M_train_joint_3d.json
  skeleton.txt
```

每帧读取：

```text
right.pose  [48]
right.shape [10]
right.trans [3]
left.pose   [48]
left.shape  [10]
left.trans  [3]
```

第一版过滤规则：

- 只取 `hand_type = right` 或单手帧；
- 只取 MANO 参数非空的帧；
- 过滤 interacting hand，避免双手贴合导致网格/声场复杂；
- 从 pose 中筛选拇指-食指距离较近的 OK/pinch 类姿态；
- 从其余姿态中采样张手、半握、握拳、指向等类别；
- 对连续序列降采样，避免大量近重复姿态。

OK/pinch 筛选可用：

```text
distance(thumb_tip, index_tip) < 20 mm
distance(thumb_tip, middle_tip) > distance(thumb_tip, index_tip)
index flexion 较大
thumb opposition 较大
```

这些条件基于 MANO 生成后的 21 关键点或 mesh tip 点计算。

## 对我们数据生成的影响

网络输入建议改成：

```text
root_translation_m [3]
root_rotation_6d   [6]
mano_pose          [48]
target_world_m     [3]
```

如果固定 shape：

```text
mano_shape beta = beta0
```

不作为网络输入。

如果想覆盖不同人的手型：

```text
mano_shape beta [10]
```

也可作为输入，但这会显著增加学习难度，不建议第一版加入。

## 近期实施任务

1. 已接入 `smplx` MANO Python layer。
2. 下载 MANO model 文件并放到本地私有路径，不提交模型文件。
3. 下载 InterHand2.6M annotation-only。
4. 已写 `scripts/hand_dataset/preview_mano_pose.py`：
   - 读取一帧 MANO 参数；
   - 生成 mesh；
   - 可视化 mesh 和 21 关键点；
   - 支持筛选 pinch/OK 姿态。
5. 已写 `scripts/hand_dataset/run_mano_hand_bem_demo.py`：
   - 读取 `preview_mano_pose.py` 导出的 MANO VTK；
   - 作为刚性障碍物加入 BEM；
   - 输出 256 相位和振幅 level set。
6. 用真实 MANO OK/pinch 姿态替换当前过程性 rig demo。

## 当前 MANO 使用命令

先从 MANO 官网下载模型文件：

```text
https://mano.is.tue.mpg.de/
```

本地建议放置：

```text
models/mano/MANO_RIGHT.pkl
models/mano/MANO_LEFT.pkl
```

这些文件受 MANO license 约束，已加入 `.gitignore`，不要提交。

使用模板零姿态做 smoke test：

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\preview_mano_pose.py `
  --mano-model-dir models\mano `
  --params-json configs\mano\zero_pose_right.json `
  --output-dir outputs\mano_pose_preview
```

使用 InterHand2.6M 的一帧 MANO 参数：

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\preview_mano_pose.py `
  --mano-model-dir models\mano `
  --interhand-mano-json data\InterHand2.6M\annotations\train\InterHand2.6M_train_MANO_NeuralAnnot.json `
  --capture-id 0 `
  --frame-idx 123 `
  --hand right `
  --output-dir outputs\mano_pose_preview
```

将 MANO mesh 接入 BEM 聚焦：

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\run_mano_hand_bem_demo.py `
  --mano-scene-mesh outputs\mano_pose_preview\mano_scene_hand.vtk `
  --output-dir outputs\mano_hand_bem_demo
```

如果没有 MANO 模型文件，脚本会提示需要放置的 `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` 路径。

## 来源

- MANO official: https://mano.is.tue.mpg.de/
- InterHand2.6M official: https://mks0601.github.io/InterHand2.6M/
- FreiHAND official: https://lmb.informatik.uni-freiburg.de/projects/freihand/
- HanCo official: https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html
- DexYCB official: https://dex-ycb.github.io/
- HOT3D official: https://facebookresearch.github.io/hot3d/
- OakInk official: https://oakink.net/
