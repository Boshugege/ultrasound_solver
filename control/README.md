# SonicSurface 本地控制服务

本目录提供独立的客户端/服务端控制链路。服务端独占串口并把收到的
256 通道相位发送给 SonicSurface；客户端只负责读取相位结果并通过 HTTP
提交，不直接访问硬件。

实时 PDE 相位的持久二进制数据面设计与分阶段 Todo 见
[`docs/REALTIME_PHASE_STREAMING_PLAN.md`](../docs/REALTIME_PHASE_STREAMING_PLAN.md)。

## 修正后的 PDE 求解器

当前 `front_focus_alg.py` 默认使用
`src/examples/pde_homogeneous_16x16.yaml`：
官方参考坐标 + `sinc_neumann` 有限孔径通量源 + 均匀介质有限体积 PDE。
阵列底面为反射挡板，其余五面使用开放边界和 PML；默认网格为 1/3 mm。
相位量化遵循官方截断规则，保留官方坐标表中的约 2 mm 原点偏移。
配置的物理后端仍为原有 `cudss_hybrid_direct`。均匀自由场使用
内置表面凝聚算法加速同一个 PDE 的目标响应，不存在第二套物理后端。

`front_focus_alg.py`、`circle.py`、`line.py` 和 `cube.py` 均使用这一配置。
官方传播公式仅用于离线对照脚本，不再注册为项目求解后端。

```powershell
conda activate srt
python control/front_focus_alg.py --offline
```

该命令不访问真实硬件；`front_focus_alg.py` 未带 `--offline`
时仍执行原有发射流程。
PDE 目标响应缓存在 `outputs/pde_homogeneous_16x16/target_responses.npz`，
缓存指纹覆盖网格、几何、介质、声源、边界、PML 和实现代码。
场单位为归一化 `1/m`，不能解释成经过硬件标定的绝对 Pa。
物理修改和代码整理范围见 `docs/PHYSICAL_MODEL_ALIGNMENT_AND_BACKEND.md`。

```text
逆向优化结果 (.npy/.npz)
        |
        v
control/client.py
        |  HTTP JSON
        v
control/server.py
        |  映射 -> 校正 -> 32 档量化 -> UART
        v
SonicSurface FPGA
```

## 快速使用（Conda srt + COM5）

以下命令均在仓库根目录 `c:\workspace\ultrasound_solver` 执行。

首先激活环境并确认串口可见：

```powershell
conda activate srt
python -m serial.tools.list_ports
```

确认输出中存在 `COM5`。然后检查 `control/sonicsurface.example.yaml`：

```yaml
serial:
  port: "COM5"
  baudrate: 230400

channel_order: "sonicsurface"

calibration:
  mode: "firmware"
  phase_corrections_bins: "zeros"
  verified: true
```

只有确认板卡烧录的是仓库内已校准的
`thirdparty/SonicSurface/Firmware/FPGA primary/output_file.jic` 时，才能使用
上述 `firmware + verified: true`。如果烧录的是
`FPGAPrimaryNoCalib.jic`，必须先完成标定，并改用主机校正或重新编译固件。

打开终端 1，启动实际串口服务：

```powershell
conda activate srt
python control\server.py --config control\sonicsurface.example.yaml --live
```

打开终端 2，先检查状态并发送关闭命令：

```powershell
conda activate srt
python control\client.py status
python control\client.py off
```

发送全阵元零相位测试：

```powershell
python control\client.py uniform 0
```

以 `200 Hz` 波群频率、`50%` 占空比调制零相位图案，持续 3 秒：

```powershell
python control\client.py modulate `
  --uniform-phase 0 --frequency 200 --duty 0.5 --duration 3
```

发送逆向优化结果文件：

```powershell
python control\client.py send `
  outputs\<场景>\<算法>\result_best_phases_rad.npy
```

也可以直接指定结果目录，客户端会优先选择最佳相位文件：

```powershell
python control\client.py send outputs\<场景>\<算法>
```

以指定波群频率调制逆向优化结果：

```powershell
python control\client.py modulate `
  outputs\<场景>\<算法> --frequency 200 --duty 0.5
```

不提供 `--duration` 时会持续调制，直到执行 `off`、发送新的静态图案或启动
另一组调制。这里调整的是作用于 40 kHz 超声载波的包络/波群频率，不会改变
换能器的 40 kHz 载波频率。

周期扫描一段频率并实时显示当前值：

```powershell
python control\frequency_sweep.py
```

运行前直接修改 `control/frequency_sweep.py` 顶部的实验变量：

```python
SERVICE_URL = "http://127.0.0.1:8765"
MIN_FREQUENCY_HZ = 20.0
MAX_FREQUENCY_HZ = 300.0
SWEEP_PERIOD_S = 10.0
DUTY_CYCLE = 0.5
DURATION_S = None
REPORT_INTERVAL_S = 0.2
```

其中 `DURATION_S = None` 表示持续运行，直到按 `Ctrl+C`；也可以改为
`60.0`，表示扫描 60 秒后自动停止。程序默认使用：

- 场景配置：`src/examples/pde_homogeneous_16x16.yaml`
- 相位文件：
  `outputs/pde_homogeneous_16x16/front_focus_alg_phases.npz#phases_rad`
- 目标点：阵列中心正前方 `[0.08, 0.08, 0.12] m`
- 边界：板面 `-z` 保持 `reflecting`，其余五面 `open`
- 障碍物：无

相位文件不存在时，程序会先调用当前 Conda 环境的 Python 运行 PDE
目标响应优化并生成文件。扫描采用三角波规律：频率从最低值连续升至最高值，再降回
最低值；`SWEEP_PERIOD_S` 表示一次完整往返的时间。终端持续输出当前包络
频率和已完成周期数。按 `Ctrl+C` 会请求服务端停止扫描并关闭全部阵元。

实验结束后关闭全部阵元：

```powershell
python control\client.py off
```

## 圆周运动 Demo

`control/circle.py` 将焦点放在与阵列平行的圆周上：

- 阵列边长：`0.16 m`
- 圆心：阵列正中央 `[0.08, 0.08] m`
- 高度：`z = 0.06 m`
- 半径：阵列边长的二十分之一，即 `0.008 m`
- 离散点数：16
- 播放速度：1 帧/秒

参数集中在 `control/circle.py` 顶部：

```python
BOARD_SIDE_M = 0.16
TARGET_HEIGHT_M = 0.06
CIRCLE_RADIUS_M = BOARD_SIDE_M / 20.0
PATH_POINT_COUNT = 16
OPTIMIZATION_ITERATIONS = 120
PLAYBACK_HZ = 1.0
LOOP_PLAYBACK = True
ENVELOPE_FREQUENCY_HZ = None
ENVELOPE_DUTY_CYCLE = 0.5
```

运行：

```powershell
conda activate srt
python control\circle.py
```

首次运行会：

1. 使用 `CirclePath` 将圆周离散成 16 个目标点。
2. 使用修正后的均匀场有限体积 PDE 计算目标响应。
3. 对每个目标点独立优化相位，并使用前一目标点相位热启动。
4. 将所有目标点和对应的 256 通道相位保存到：
   `outputs/circle_front_16x16/circle_phases.npz`。
5. 连接服务端并依次发送相位，使焦点沿圆周循环移动。

圆周运动默认直接覆盖相位，不在相邻点之间关闭阵列。若显式启用
`ENVELOPE_FREQUENCY_HZ`，每次换点都必须重建调制缓冲区，不适合平滑运动。
固定点触觉测试可使用 `front_focus_alg.py` 的包络模式。

相位缓存已经存在且参数未改变时，之后运行会直接进入播放，不再计算。修改
路径参数、优化配置或算法/几何实现后缓存会自动失效；也可将 `FORCE_RECOMPUTE` 改为 `True`
强制重新计算。按 `Ctrl+C` 会停止播放并关闭全部阵元。

## 小球圆周搬运 Demo

`control/cotton_ball_circle.py` 搬运阶段采用连续 40 kHz 聚焦，不使用触觉
开关包络。程序启动后先在阵列中心上方保持静态焦点。此时可按 `L` 临时发射
2 秒 `200 Hz` 包络来用手定位，随后自动恢复连续中心场；放置轻质颗粒后直接
按数字键 `0` 启动运动，无需回车。随后焦点先从圆心平滑移动到圆周，再沿
圆周循环：

```powershell
python control\cotton_ball_circle.py
```

默认参数：

```text
中心             (0.08, 0.08, 0.06) m
圆周半径         0.01 m
中心到圆周       32 点，2 秒
圆周             96 点，8 秒/圈
播放方式         连续直接覆盖相位，无中间关闭帧
```

可通过命令行调整：

```powershell
python control\cotton_ball_circle.py `
  --height 0.06 `
  --radius 0.01 `
  --transfer-points 32 `
  --transfer-seconds 2 `
  --circle-points 96 `
  --circle-seconds 8
```

使用 `--once` 只运行一圈，使用 `--force-recompute` 忽略相位缓存。生成的相位
保存在：

```text
outputs/cotton_ball_circle_16x16/cotton_ball_circle_phases.npz
```

运动阶段只请求 `/v1/pattern`。服务端每次发送
`0xFE + 256 相位字节 + 0xFD`，FPGA 在完整相位帧接收后直接交换缓冲区；
相邻目标点之间不会插入全关闭图案，因此不会出现由关闭/重启造成的卡顿。

连续 40 kHz 的稳态中心场本来就很难由手直接感觉，`L` 定位模式只用于确认
大致坐标，不能承托颗粒。放置时应按坐标使用塑料镊子、细网勺或非金属高度
夹具送入，不要依靠手感寻找位置。

普通棉花球吸声、形状不规则且通常过大，不是合适的首个悬浮样品。应优先使用
直径约 1--3 mm、质量极轻且规则的 EPS/聚苯乙烯泡沫小球。当前脚本采用单焦点，
不保证形成三维稳定势阱；若中心连续场不能稳定承托颗粒，圆周运动也无法可靠
搬运，需要改用 twin/vortex/bottle 陷阱或双阵列驻波结构，而不是继续调整轨迹。

## PDE 固定涡旋测试

`control/vortex.py` 使用当前 sinc Neumann PDE 的目标响应，在目标中心周围
构造一阶复相位绕转，并在优化中同时最大化环形涡旋模态、抑制中心残余压力。
它不是将官方几何聚焦相位直接加上方位角。

先运行离线验证：

```powershell
python control\vortex.py --offline
```

默认目标中心为 `(0.08, 0.08, 0.06) m`，验证环半径为 `5 mm`，使用 32 个
环采样点。当前默认结果：

```text
连续相位：中心/环均值 0.471%，模式纯度 99.962%
32 档相位：中心/环均值 1.510%，模式纯度 99.960%
```

真实固定点测试：

```powershell
python control\vortex.py
```

涡旋保持期间可按 `L`，程序会临时切换到同一坐标的 PDE 单焦点并施加
`200 Hz` 包络 2 秒，便于用手确认位置，然后自动恢复涡旋。定位期间不是
涡旋陷阱，不应放置颗粒。也可使用 `--duration 10` 限制总发射时间。

正常保持阶段通过 `/v1/pattern` 发送涡旋相位并连续保持 40 kHz，不重复写
串口。相位和指标保存在 `outputs/vortex_fixed_16x16/vortex_phases.npz`。

上述指标证明离散声压场具有中心低压核与一阶相位绕转，但尚未计算特定颗粒的
Gor'kov 势、三维声辐射力和陷阱刚度，因此不能仅凭此宣称稳定悬浮。应先使用
1--3 mm EPS 泡沫球验证静态捕获，再考虑移动涡旋。

## 直线前后运动 Demo

`control/line.py` 让焦点沿阵列中心法线往返移动：

```text
0.12 m → 0.03 m → 0.12 m → ...
```

当前 PDE 实现：

```powershell
conda activate srt
python control\line.py `
  --point-count 16 `
  --round-trip-seconds 6 `
  --start-height 0.12 `
  --end-height 0.03
```

官方 `SonicSurface.phasesForFocus()` 对照实现：

```powershell
python control\line_official.py `
  --point-count 16 `
  --round-trip-seconds 6 `
  --start-height 0.12 `
  --end-height 0.03
```

- `--point-count`：线段上的唯一振幅点数，包含两个端点。
- `--round-trip-seconds`：起点到终点再返回起点的完整往返时间。
- `--start-height`、`--end-height`：竖直线段两端高度，单位米。
- `--once`：只往返一次；缺省时循环播放。
- `--envelope-frequency`：显式启用触觉包络频率；轨迹默认不启用。
- `--no-envelope`：关闭每点重新启动的包络，只连续切换聚焦相位。
- PDE 版本额外支持 `--force-recompute` 和 `--config`。

程序只计算 16 个唯一高度点，播放时按正序和反序组成 30 个播放点，避免在
两个端点重复停顿。两条对照命令必须使用完全相同的参数。结果分别保存为：

```text
outputs/line_front_16x16/line_phases.npz
outputs/line_front_16x16/line_phases_official.npz
```

之后直接读取缓存并发送。设置 `ENVELOPE_FREQUENCY_HZ = None` 可关闭包络，
设置 `FORCE_RECOMPUTE = True` 可强制重新计算全部高度点。

若 PDE 和官方版本都出现杂斑，先进行相同参数的无包络对照：

```powershell
python control\line.py --round-trip-seconds 3 --no-envelope
python control\line_official.py --round-trip-seconds 3 --no-envelope
```

有包络时，每次换点会停止旧调制并发送旧关闭、新聚焦和新关闭三帧，同时
重新启动包络；无包络时每点只发送一帧聚焦相位。若无包络模式明显平滑，
杂斑主要来自逐点重启包络和串口时序，而不是 PDE/官方相位差异。若无包络
仍存在多个稳定触点，则应检查硬件相位校准、阵元极性和阵列旁瓣。

## 横向直线运动 Demo

横向版本让焦点在固定高度沿阵列 `x` 方向往返，手掌保持覆盖阵列中心上方
即可感受焦点经过，无需跟随焦点上下移动。默认路径为：

```text
(0.05, 0.08, 0.06) m ↔ (0.11, 0.08, 0.06) m
```

默认使用 24 个唯一位置、46 个往返播放点和 3 秒往返时间。相邻位置约
`2.6 mm`，播放更新率约 `15.3 点/秒`，相邻点之间直接覆盖相位。

当前 PDE 实现：

```powershell
python control\horizontal_line.py `
  --point-count 24 `
  --round-trip-seconds 3 `
  --start-x 0.05 `
  --end-x 0.11 `
  --height 0.06
```

官方算法对照：

```powershell
python control\horizontal_line_official.py `
  --point-count 24 `
  --round-trip-seconds 3 `
  --start-x 0.05 `
  --end-x 0.11 `
  --height 0.06
```

两条命令还支持 `--once`、`--envelope-frequency` 和 `--no-envelope`；PDE
版本额外支持 `--force-recompute` 和 `--config`。相位结果分别保存为：

```text
outputs/horizontal_line_16x16/horizontal_line_phases.npz
outputs/horizontal_line_16x16/horizontal_line_phases_official.npz
```

## 立方体十二棱运动 Demo

`control/cube.py` 让焦点依次经过空间立方体的全部 12 条棱。默认立方体位于
阵列中心上方，边长为 `0.04 m`，中心高度为 `0.075 m`。每条棱的点数和
一次完整遍历的时间可直接通过命令行设置：

```powershell
conda activate srt
python control\cube.py `
  --points-per-edge 6 `
  --traversal-seconds 4
```

- `--points-per-edge`：每条棱包含的振幅点数，包含两个端点，最少为 2。
- `--traversal-seconds`：遍历全部 12 条棱所需的目标时间，单位秒。
- `--edge-length`：立方体边长，单位米，默认 `0.04`。
- `--center-height`：立方体中心高度，单位米，默认 `0.075`。
- `--once`：只遍历一次；缺省时循环播放。
- `--force-recompute`：忽略缓存并重新计算全部相位。

立方体角点会被多条棱共享，但程序只为同一空间点计算一次 PDE 相位。默认
参数生成 56 个唯一求解点和 64 个实际播放点。由于立方体的 8 个顶点都是
奇数度顶点，不存在恰好一次连续走完全部 12 条棱的路径；程序将边组织为
最少的 4 段连续轨迹，段间进行 3 次焦点跳转。

立方体轨迹默认直接覆盖相位，每个点只发送一帧 `258` 字节，不在相邻点之间
关闭阵列。程序会根据点数和遍历时间自动计算更新率，并拒绝超过
`230400 baud` 串口理论带宽的设置。首次运行生成：

```text
outputs/cube_edges_16x16/cube_edge_phases.npz
```

使用官方 `SonicSurface.phasesForFocus()` 生成同一条轨迹，可运行：

```powershell
python control\cube_official.py `
  --points-per-edge 6 `
  --traversal-seconds 4 `
  --edge-length 0.04 `
  --center-height 0.075
```

比较 PDE 与官方版本时，两条命令必须使用完全相同的参数。官方版本仍复用
相同的路径离散、通道映射、32 档量化、直接相位切换和串口服务，因此两次实验的
主要差异只有聚焦相位的生成方法。官方相位另存为：

```text
outputs/cube_edges_16x16/cube_edge_phases_official.npz
```

若两个版本都有相同杂点，应优先检查 12 条棱之间的 3 次不可避免跳转、目标点
切换速率、相位校准、阵元极性和硬件旁瓣；若仅 PDE 版本明显出现杂点，再比较
两个 NPZ 中同一目标点的相位。

## 文件说明

```text
control/
  client.py                     客户端入口
  circle.py                     圆周路径预计算与循环播放
  cotton_ball_circle.py         中心等待后直接切相位的轻球圆周搬运
  cube.py                       立方体十二棱路径预计算与循环播放
  cube_official.py              官方算法立方体十二棱对照版本
  frequency_sweep.py            包络频率周期扫描
  front_focus_alg.py            PDE 正前方聚焦与离线验证
  horizontal_line.py            PDE 横向直线往返路径
  horizontal_line_official.py   官方算法横向直线往返对照版本
  line.py                       PDE 中心轴竖直直线往返路径
  line_official.py              官方算法竖直直线往返对照版本
  server.py                     服务端入口
  sonicsurface.example.yaml     配置模板
  vortex.py                     PDE 固定一阶声学涡旋测试
  sonicsurface/
    protocol.py                 通道映射、量化和帧编码
    phase_source.py             求解器相位结果加载
    controller.py               dry-run/串口控制器
    server.py                   HTTP 服务
    client.py                   客户端命令
  thirdparty/SonicSurface/      官方 SonicSurface 仓库
```

## 协议依据

实现直接依据：

- `thirdparty/SonicSurface/ControlSoftware/Python/SonicSurface.py`
- `thirdparty/SonicSurface/MyPatterns/my_patterns.py`
- `thirdparty/SonicSurface/Firmware/FPGA primary/src/Distribute.vhd`

板卡使用 `230400 baud, 8N1`。一帧共 258 字节：

```text
0xFE + 256 个相位字节 + 0xFD
```

相位字节 `0..31` 对应一个周期的 32 个离散档位，`32` 表示关闭通道。
`0xFD` 交换 FPGA 双缓冲，使整幅相位同时生效。

## 1. 准备配置

复制 `control/sonicsurface.example.yaml` 后修改。默认配置：

- 使用官方 `EMITTERS_ORDER` 通道映射。
- 使用 FPGA 固件内置校正。
- `calibration.verified` 为 `false`，禁止真实串口发送。

校正只能存在一处：

- `mode: firmware`：FPGA 的 `PHASE_CORRECTION` 已写入校正，主机不再补偿。
- `mode: host`：用于无校正固件，填写 256 项
  `phase_corrections_bins`，服务端发送前补偿。

第三方仓库的 `FPGAPrimaryNoCalib.jic` 明确是不含校正的预编译固件。不能在
不确认当前固件的情况下把配置标记为已验证。

## 2. Dry-run 联调

先启动服务端。此命令不会打开串口：

```powershell
python control\server.py `
  --config control\sonicsurface.example.yaml
```

另开终端查询状态：

```powershell
python control\client.py status
```

发送全阵元零相位测试帧：

```powershell
python control\client.py uniform 0
```

关闭全部通道：

```powershell
python control\client.py off
```

dry-run 返回 `"accepted": true` 和 `"transmitted": false`，说明数据已通过
校验并成功编码，但没有访问硬件。

## 3. 波群频率调制

当前已校准的 primary 固件不支持触觉调制固件的 `101xxxxx` 调频命令。
本服务采用官方示例 `TestBeamStearingWithAudibleMod.py` 和
`MyPatterns/my_patterns.py` 使用的双缓冲方案：

1. 将目标相位和全关闭图案预装进 FPGA 的两个缓冲区。
2. 服务端按指定频率发送单字节 `0xFD`，在两个缓冲区之间切换。
3. 调制结束、收到 `off` 或发生异常时，服务端发送全关闭帧。

支持范围：

- `--frequency`：`1..500 Hz`，默认 `200 Hz`。
- `--duty`：`0.05..0.95`，默认 `0.5`。
- `--duration`：可选，单位秒；省略表示持续运行。

Windows 上的软件定时不是硬实时。需要严格稳定的高频包络时，应改用
`FPGATactileModulation.jic` 及其 FPGA 内部调制命令，但该固件的相位校准值
必须针对当前阵列重新确认。

## 4. 发送逆向优化结果

直接指定推荐的最优相位：

```powershell
python control\client.py send `
  outputs\<场景>\<算法>\result_best_phases_rad.npy
```

也可以指定包含标准结果的目录。客户端按以下顺序自动选择：

1. `result_best_phases_rad.npy`
2. `result_final_phases_rad.npy`
3. `result.npz` 中的 `best_phases`
4. `result.npz` 中的 `phases`

```powershell
python control\client.py send outputs\<场景>\<算法>
```

指定 NPZ 键：

```powershell
python control\client.py send result.npz --npz-key best_phases
```

所有输入必须是单位为弧度、形状为 `(256,)` 的有限浮点数组。旧的 12×12
结果形状为 `(144,)`，客户端会明确拒绝，不能直接发送到 16×16 板卡。

## 5. 真实连接板卡

1. 用 USB 转串口连接电脑与控制板，确认公共地线、电平和供电正确。
2. 在 Windows 设备管理器确认串口，例如 `COM7`。
3. 确认 FPGA 已烧录 SonicSurface 兼容固件。
4. 确认固件是否内置相位校正，并按实际情况设置 `calibration.mode`。
5. 验证 256 通道顺序和校正结果。
6. 将配置中的 `serial.port` 改为实际端口。
7. 只有完成上述检查后，设置 `calibration.verified: true`。
8. 使用 `--live` 启动服务：

```powershell
python control\server.py `
  --config control\sonicsurface.local.yaml `
  --live
```

`/v1/pattern` 默认不等待串口：服务端把帧交给后台写线程后立即返回，写线程在
串口空闲时总是发送最新的一帧。`230400 baud` 下一帧需要 `11.2 ms`，上限约
`89 帧/秒`；客户端发送得更快时，尚未写出的旧帧会被新帧替换，不会排队，
因此发出的永远是最新相位。响应中的 `transmitted: true` 表示已交给真实串口
（dry-run 为 `false`），`queued: true` 表示走的是这条异步路径；需要等帧写完
再返回时，在请求体中加 `"wait": true`。`/v1/off` 和 `/v1/modulation` 始终同步
写出，并先丢弃尚未写出的帧，因此关闭命令不会被之前的相位覆盖。

客户端（`sonicsurface.client.request_json`，各 demo 脚本都经由它）连接本机服务时
先不带超时建立连接，连上后再设置收发超时：Windows 上带超时的 `connect()` 即使
连 `127.0.0.1` 也常多等约 16 ms，原先每帧新建这样的连接，加上同步等待串口写完，
模拟测试中只能跑到约 42 帧/秒。服务端同时打开了 `TCP_NODELAY`。

`/v1/status` 的 `stream` 字段给出写线程统计：`frames_queued`、`frames_written`、
`frames_superseded`（被新帧替换或被关闭丢弃的帧）、`write_errors`、
`last_write_ms`/`max_write_ms`（单帧写串口耗时）、`last_latency_ms`（从提交到
写完）以及 `pending`。逐帧日志在流式发送时会刷屏，因此改为 DEBUG 级别，需要时
用 `--log-level DEBUG` 打开，日志会记录每次相位提交的来源、是否真实发送和帧
SHA-256。停止实验前先执行：

```powershell
python control\client.py off
```

若日志出现 `SerialTimeoutException: Write timeout`，说明 USB 串口驱动或板端
接收已阻塞，不是 PDE 求解失败。当前控制器对带 `0xFE` 帧头的完整相位帧会
自动关闭并重开串口，再完整重发一次；该帧头会重置 FPGA 字节计数器，因此
不会把半帧继续当作相位。单字节双缓冲交换命令不会盲目重试，以免不确定地
交换两次。状态接口中的 `serial_recoveries` 和 `last_serial_error` 可用于诊断。

移动轨迹默认使用 `/v1/pattern` 直接覆盖相位。不要在每个轨迹点重新启动
`/v1/modulation`；这会反复发送关闭缓冲区并增加串口超时和运动卡顿风险。

## 局域网使用

默认只监听 `127.0.0.1`。若客户端在另一台可信电脑上，可显式监听局域网：

```powershell
python control\server.py `
  --config control\sonicsurface.local.yaml `
  --host 0.0.0.0 --live
```

客户端指定服务端地址：

```powershell
python control\client.py `
  --service-url http://192.168.1.20:8765 status
```

当前接口没有身份认证，不应暴露到公网。需要跨机器使用时，应通过 Windows
防火墙只允许可信局域网主机访问端口 `8765`。

## HTTP 接口

```text
GET  /v1/status
POST /v1/pattern
POST /v1/modulation
POST /v1/frequency-sweep
POST /v1/off
```

`/v1/pattern` 请求：

```json
{
  "phases_rad": [0.0, 0.1, 0.2],
  "source": "outputs/example/adjoint/result_best_phases_rad.npy",
  "wait": false
}
```

实际数组必须包含 256 项。数值表示弧度；`null` 可用于关闭单个通道。`wait`
可省略，默认 `false`（见第 5 节）。

`/v1/modulation` 在相同相位数组之外接受：

```json
{
  "phases_rad": [0.0, 0.1, 0.2],
  "frequency_hz": 200.0,
  "duty_cycle": 0.5,
  "duration_s": 3.0,
  "source": "result_best_phases_rad.npy"
}
```

实际数组同样必须包含 256 项。`duration_s` 可以省略，此时调制持续到
`POST /v1/off` 或下一次相位提交。

`/v1/frequency-sweep` 使用三角波周期改变包络频率：

```json
{
  "phases_rad": [0.0, 0.1, 0.2],
  "min_frequency_hz": 20.0,
  "max_frequency_hz": 300.0,
  "sweep_period_s": 10.0,
  "duty_cycle": 0.5,
  "duration_s": 60.0,
  "source": "result_best_phases_rad.npy"
}
```
