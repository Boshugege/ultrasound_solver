训练神经网络近似：

```text
MANO 手型 + 手部姿态 + 手根位姿 + 目标点
→ 256 通道复声学响应
```

单条数据

```text
shape_beta           float32 [10]
hand_pose            float32 [45]
root_rotation        float32 [3]
root_translation_m   float32 [3]
target_world_m       float32 [3]
response_real        float32 [256]
response_imag        float32 [256]
```

## 采集流程

1. 从 FreiHAND 姿态和合法扰动中采样 MANO `pose`，并采样多组 `shape_beta`。
2. 生成闭合 MANO 网格，拒绝穿模、越界和非闭合样本。
3. 每个姿态在阵列上方采样大量目标点；覆盖均匀空间点、手部附近点和连续轨迹点。目标必须位于手外。
4. 使用 BEM 计算每个目标点对应的 256 通道复响应。
5. 保存输入、标签、求解器配置、收敛状态。

## 数据划分

- `Train / Validation / Test-ID`：按手型和姿态簇划分。
- `Test-OOD`：使用训练中未出现的手型、姿态范围或目标区域。

<br />

## 建议规模

- Pilot：500 个几何状态 × 64 个目标点，约 3.2 万条。
- MVP：8000 个几何状态 × 128 个目标点，约 102 万条。
- Full：验证误差仍随数据量明显下降时，再扩大到约 1280 万条。

几何状态指一组确定的 `shape_beta + hand_pose + root pose`。

## 当前耗时估算

教师网格：MANO 原生网格只有 1552 个面元，平均边长约 8.5 mm，接近一个波长（8.6 mm）。
在这个网格上算出的标签，聚焦效率中位数只有 67–76%，比完全忽略手（82–85%）还差。

因此打标签前先把手均匀重划分为边长约 λ/4 的三角形，约 2.1–2.4 万个面元
（`--elements-per-wavelength 4`，默认值）。以 λ/5 为参考：

- 标签误差中位数 2.5–3.1%；
- 聚焦效率中位数 99.93%，P5 99.5–99.7%。

对比见 `python bem_study/teacher_labels.py resolution --mano 3 --uniform 3 4 5`。

RTX 4090 实测（High 教师、128 目标，所有目标点共用一次矩阵分解）：

- 每个几何状态约 2.3–3.1 秒，含重划分；
- MVP（8000 × 128）：约 6 小时；
- Full（50000 × 256）：约 2 天，按每个几何状态约 3–3.5 秒估算。

耗时随面元数变化；同一输出目录只允许一个生成进程。

## 启动采集

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\generate_mano_bem_dataset.py `
  --output-dir outputs\hand_phase_surrogate_pilot_v1 `
  --num-geometries 500 --targets-per-geometry 64 --num-shapes 10
```

程序按几何状态原子保存，重复执行同一命令会自动跳过已完成样本。进度记录在 `outputs/hand_phase_surrogate_pilot_v1/progress.json`。

MVP 使用 High 教师配置：

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\generate_mano_bem_dataset.py `
  --output-dir outputs\hand_phase_surrogate_mvp_v1 `
  --num-geometries 8000 --targets-per-geometry 128 --num-shapes 64 `
  --teacher-profile high
```

进度记录在 `outputs/hand_phase_surrogate_mvp_v1/progress.json`。

Full 数据集规模为 50000 个几何状态、每状态 256 个目标点、256 种手型，共约 1280 万条。当前使用以下脚本在 MVP 完整成功后自动启动，并在启动前检查至少 35 GB 可用空间：

```powershell
.\scripts\hand_dataset\run_full_after_mvp.ps1 `
  -WaitForProcessId <MVP主进程PID> `
  -PythonExe C:\Users\Admin\miniconda3\envs\srt\python.exe
```
