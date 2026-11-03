# MANO 多手势驱动与 BEM 聚焦演示

当前流程从 FreiHAND 的 32,560 组 MANO 参数中筛选真实姿态，已生成四组独立结果：

| 手势 | FreiHAND 样本 | BEM 面元 | 40% level set 三角形 |
|---|---:|---:|---:|
| OK | 3325 | 1552 | 16944 |
| 张开手掌 | 0 | 1552 | 6936 |
| 握拳 | 3772 | 1552 | 5954 |
| 食指指向 | 1125 | 1552 | 18806 |

统一交互查看程序：

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\view_mano_gesture_suite.py --gesture ok
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\view_mano_gesture_suite.py --gesture open
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\view_mano_gesture_suite.py --gesture fist
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\view_mano_gesture_suite.py --gesture point
```

每组输出位于 `outputs/mano_gestures/<gesture>/`，其中 `pose/` 保存 MANO 三视图和闭合网格，`bem/` 保存相位、振幅体、level set、报告与场景配置。交互程序支持鼠标旋转、缩放、平移以及振幅阈值滑条。

level set 采样会屏蔽障碍物内部及距表面小于 1 mm 的区域，避免 BEM 近表面数值尖峰错误主导全局峰值。

## 旧过程性 Rig 演示

该 demo 用于验证当前链路：

```text
固定手网格 -> 过程性关节驱动 -> OK 手势变形网格
-> BEM 目标点复响应 -> 256 通道相位 -> 局部振幅场 level set
```

运行命令：

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\run_ok_hand_bem_demo.py
```

交互式查看器：

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\view_ok_hand_bem_demo.py
```

查看器功能：

- 鼠标旋转、缩放、平移 3D 场景；
- 显示 OK 手势半透明网格；
- 显示 256 个换能器，并按 BEM 聚焦相位周期着色；
- 显示红色目标点；
- 显示当前振幅场等值面；
- 底部滑条可调等值面阈值，拖动时复用已采样振幅场重新提取 marching-cubes 等值面。

离屏截图验证：

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\view_ok_hand_bem_demo.py `
  --screenshot outputs\hand_ok_bem_demo\interactive_viewer_screenshot.png
```

默认目标点：

```text
[0.08, 0.08, 0.065] m
```

主要输出：

```text
outputs/hand_ok_bem_demo/
  ok_hand_scene.vtk
  ok_hand_bem_scene.yaml
  ok_hand_bem_response.npz
  ok_hand_levelset.npz
  ok_hand_bem_visualization.png
  interactive_viewer_screenshot.png
  ok_hand_bem_report.html
  metrics.json
```

当前验证结果：

```text
BEM hand elements: 2368
target response time: about 18 s
level set threshold: 40% peak
level set triangles: 132
```

## 说明

当前手部驱动是自动估计的过程性 rig：

- 从 `hand_watertight.vtk` 读取固定闭合网格；
- 自动聚类估计 5 个手指区域；
- 使用连续权重形变和拇指/食指指尖吸引生成 OK pinch 手势；
- 保留原始网格拓扑和面元顺序；
- 输出闭合 VTK 网格供 BEM 使用。

交互视图中目标点使用浅色半透明小球和红色中心点，避免遮挡 cyan 振幅等值面；判断聚焦是否正确时应看等值面是否包围该红色中心点，而不是看目标球本身。

这已经能跑通“关节参数到 BEM 相位和振幅场”的完整流程，但还不是最终训练集用的人工校准手 rig。正式数据集仍应补充：

- 21 关节骨架；
- 解剖关节自由度；
- 关节角限位；
- 人工或半自动蒙皮权重；
- 自碰撞和穿模检查。

当前 demo 的 BEM 设置偏向流程验证：

```text
bem_rtol: 1e-2
bem_elements_per_wavelength: 2
source quadrature: 4 x 16 / 8 x 32
```

正式训练标签不要直接使用该配置，应使用更高精度教师配置。
