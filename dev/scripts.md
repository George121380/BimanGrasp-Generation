# 运行指令说明（BimanGrasp-Generation）

## 环境与安装
- 激活环境（已安装情况下）
  ```bash
  conda activate bimangrasp
  ```
- 一键安装（可选）
  ```bash
  bash install.sh
  ```

## 基础运行（生成抓取）
- 推荐从仓库根目录运行（或使用绝对路径运行）。
- 单对象、批量样本、固定迭代数：
  ```bash
  python BimanGrasp-Optimization/main.py \
    --name exp_run \
    --object_code Curver_Storage_Bin_Black_Small \
    --batch_size 32 \
    --num_iterations 2000
  ```
- 输出目录：`data/experiments/exp_run/results/`（含 `config.txt` 与每对象 `.npy` 结果）。

## 过程视频记录（可选）
- 开启视频记录、设置截帧频率、像素及多组记录：
  ```bash
  python main.py \
    --name random \
    --object_code merged_collision_300k_wt \
    --batch_size 128 --num_iterations 10000 \
    --vis --vis_frame_stride 500 --vis_fps 20 \
    --vis_width 900 --vis_height 900 \
    --vis_obj 0 --vis_local 0 --vis_record_num 16
  ```
- 输出：
  - 帧：`data/experiments/vis_run/results/frames_0/`、`frames_1/`
  - 视频：`data/experiments/vis_run/results/optimization_0.mp4`、`optimization_1.mp4`

## 指标记录（加权分量，自动绘图）
- 开启简化指标记录（记录 batch 均值的加权分量并自动绘图）：
  ```bash
  python BimanGrasp-Optimization/main.py \
    --name metrics_run \
    --object_code Curver_Storage_Bin_Black_Small \
    --batch_size 16 --num_iterations 1000 \
    --metrics --metrics_stride 50
  ```
- 输出：`data/experiments/metrics_run/results/metrics/`
  - `metrics_summary.csv`（字段：`step,total,w_dis*dis,w_pen*pen,w_spen*spen,w_joints*joints,w_vew*vew,accept_rate,temperature,step_size`）
  - `metrics_plot.png`

## 指定目标点初始化（A/B 定点、手掌朝向目标）
- 将左手初始化在 A 附近、右手在 B 附近，并使手掌朝向各自目标点（物体坐标系）：
  ```bash
  python main.py \
    --name random \
    --object_code merged_collision_300k_wt \
    --batch_size 128 --num_iterations 10000 \
    --vis --vis_frame_stride 100 --vis_record_num 16 \
    --init_at_targets \
    --left_target "0.10 0.00 0.00" \
    --right_target "-0.10 0.00 0.00" \
    --target_distance 0.32 \
    --target_jitter_dist 0.00 \
    --target_jitter_angle 0.52 \
    --target_twist_range 3.14 \
    
  ```
- 说明：A/B 为物体坐标（当前物体位姿为单位），程序内部按当前样本尺度缩放。
- 未提供 `--right_target` 时，默认镜像 `--left_target` 的 x、y 以得到 B。


## 交互模式
conda activate bimangrasp
cd BimanGrasp-Optimization
python main.py \
  --name interact_demo \
  --object_code merged_collision_300k_wt \
  --batch_size 128 --num_iterations 10000 \
  --interact --ui_port 8050 \
  --target_distance 0.25 \
  --vis --vis_frame_stride 50 --vis_record_num 16 \
  --vis_width 900 --vis_height 900



## 已有结果的可视化
- 用交互 3D 查看某个抓取：
  ```bash
  python BimanGrasp-Optimization/visualization.py \
    --object_code Curver_Storage_Bin_Black_Small \
    --num 0 \
    --result_path data/experiments/exp_run/results
  ```

## 常见问题
- 静态图导出需 `kaleido`，如遇提示请安装：
  ```bash
  pip install -U kaleido
  ```
- 若提示 Plotly/Kaleido 版本不匹配，升级 Plotly：
  ```bash
  pip install -U "plotly>=6.1.1"
  ```
- 从任意目录运行请使用绝对路径；或先 `cd BimanGrasp-Optimization`，因为 MJCF 路径是相对的。

