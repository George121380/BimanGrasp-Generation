# 交互式初始化位置选择方案（Interaction Init）

## 目标与约束
- 通过 `--interact` 开关，先进入交互页面选择两只手的初始位置（仅位置，不含朝向）；随后按所选位置初始化，再开始优化。
- 约束（初始化姿态生成时满足）：
  1. 手心朝向物体中心（approach 方向指向物体中心）。
  2. 左右两只手在“指尖朝向”上保持对称（同相位的绕掌心轴扭转）。
  3. Batch 内的随机旋转要覆盖全角度（绕掌心轴的扭转角 φ 在 [0, 2π) 近似均匀覆盖）。
- 记录：在 logs 中保存两次交互选取的坐标；在 results 中保存“物体+两个悬浮球”的可视化快照（PNG/HTML）。

## 交互方式与依赖
- 首选：Dash + Plotly Web UI（运行在 `localhost`），无窗口管理器依赖，兼容服务器；可通过鼠标/滑条精准调整。
- 备选：Open3D Visualizer（键盘/鼠标操控）或 Matplotlib 交互；若 Dash 不可用时降级到 CLI 录入。
- 新增依赖（仅交互启用时需要）：`dash`、`dash-bootstrap-components`（最小可用方案也可仅用 dash）。

## 用户流程（UI 交互）
1) `--interact` 启用时，主程序在模型加载后暂停，启动一个本地交互页面：
   - 显示当前对象网格（按样本尺度 `scale` 渲染）。
   - 显示第一个悬浮球（Left，颜色 A），支持以下交互：
     - 鼠标拖动：在当前视图平面内移动球；
     - 滑条/滚轮/按钮：沿视线（或 Z 轴）前后移动；
     - 复选项：贴附到物体表面（将球投影到最近点）；
     - 数值框：可直接输入 xyz；
     - 按钮：确认 Left 位置。
2) Left 确认后，出现第二个悬浮球（Right，颜色 B），同理交互，确认后退出交互页面。
3) 返回主程序，将两点写入 `results/interaction.json` 与 `logs/interaction.txt`，并保存可视化快照 `results/interaction_preview.{png,html}`。

## 数据与坐标系
- 交互采样的点默认为“物体坐标系下的点”（即对象未施加位姿变换时的坐标）。在渲染时将点乘以当前样本的 `scale` 显示与存储：
  - 可在 JSON 中同时记录未缩放和已缩放坐标（raw 与 scaled）。
- 若启用“贴附表面”，将点投影到网格最近点（使用 `trimesh` 最近点或现有 `ObjectModel.calculate_distance(..., with_closest_points=True)` 获取最近点/法向）。

## 初始化姿态生成（结合现有目标点初始化）
- 位置：所有样本统一使用两个用户选定的点 A（Left）、B（Right）。
- 朝向（满足约束 1）：
  - 计算物体中心 `C`（网格 AABB 中心或质心），approach 方向为 `dir_L = normalize(C - A_scaled)`、`dir_R = normalize(C - B_scaled)`；
  - 生成基础旋转 `R_align(dir)`：将手局部 +z 对齐到 `dir`（使用 Rodrigues 或现有 `_rotation_align_z_to_dir`）。
- 扭转（满足约束 2/3）：
  - 为 batch 的每个样本 `k`，采样扭转角 `φ_k ~ Uniform(0, 2π)`（或分层采样覆盖全角度）；
  - 左手 `R_L = R_align(dir_L) · Rz(φ_k) · (-hand_rot)`；右手 `R_R = R_align(dir_R) · Rz(φ_k) · hand_rot`；
  - 这样保证手心朝向目标（对齐 +z），并在绕掌心轴上使用同一 `φ_k`，从而左右手“指尖朝向”一致（对称）。
- 平移：
  - `t_L = A_scaled - d · dir_L`、`t_R = B_scaled - d · dir_R`，其中 `d = target_distance + U(-jitter_dist, +jitter_dist)`。
- 关节角：沿用现有截断正态采样（`jitter_strength`），保持与当前初始化一致。
- 接触点：可继续随机，或改为“从接近球心的候选集合中随机”提高起始质量（可选）。

## 代码改动点
- `utils/config.py`：
  - 新增 `InteractionConfig` 或将交互项加入 `InitializationConfig`：`interact (bool)`、`snap_to_surface (bool)`、`z_step (float)`、`ui_port (int)` 等轻量配置。
  - `ExperimentConfig.update_from_args` 解析 `--interact`、`--snap_to_surface` 等参数。
- `utils/interactive_select.py`（新）：
  - 函数 `select_two_points_interactive(object_model, config) -> (A_raw, B_raw, preview_paths)`：
    - 构建 Dash 应用，传入网格（vertices, faces, scale），显示 Plotly 3D 图；
    - 第一步仅显示 Left 球，提交后锁定 Left，进入 Right；
    - 按钮“贴附表面”时，将当前球位置投影到网格；
    - “保存/确认”时返回点坐标并生成 PNG/HTML 快照（`fig.write_image`/`fig.write_html`）。
  - 组件：X/Y/Z 滑条与数字输入、贴附开关、确认按钮；键盘快捷（可选）。
- `BimanGrasp-Optimization/main.py`：
  - 在 `setup_models()` 载入对象后、初始化前：若 `--interact`：
    - 调用 `select_two_points_interactive(...)` 获取 `A_raw, B_raw`；
    - 将其注入 `initialization.left_target/right_target`（字符串形式或直接张量管道），并置 `init_at_targets=True`；
    - 记录 JSON/文本日志与快照路径到 `results`。
- `utils/initializations.py`：
  - 基于已实现的 `initialize_dual_hand_at_targets(...)` 覆盖 A/B；
  - 使用新的“以物体中心为朝向、φ_k ∈ [0,2π) 分层采样”的规则（若尚未覆盖）。

## 日志与可视化记录
- 写入：`results/interaction.json`，字段：
  - `object_code`、`scale`、`left_target_raw`、`left_target_scaled`、`right_target_raw`、`right_target_scaled`、`snap_to_surface`、`timestamp`。
- 保存预览：`results/interaction_preview.png` 与 `interaction_preview.html`（Plotly 图包含物体网格以及两个颜色不同的球）。
- 控制台打印确认信息，便于审计。

## UI 操作建议（默认）
- 鼠标拖动：在视图平面内移动球；滚轮：沿视线移动；
- X/Y/Z 三轴滑条：精确控制；数值框：直接输入；
- 贴附表面：打开后球心自动吸附到最近点，避免深入物体内部；
- 自动“合法性检查”：若距离物体中心过近（`d < 0.05`），提示可能导致穿透过多；
- 两步确认：Left 确认后再调 Right，避免误触。

## 随机覆盖策略（批内 φ_k）
- 使用分层策略：`φ_k = 2π * k / batch_per_obj + ε_k`，其中 `ε_k ~ U(-π/batch, π/batch)`，使角度覆盖均匀；
- 或直接 `U(0, 2π)`，简单但均匀度略差；左右手共享同一 `φ_k` 保证“指尖朝向”一致。

## 失败与降级
- 若 Dash 无法启动（端口占用/无浏览器）：
  - 降级到非交互：在 CLI 中提示输入 A/B（或从文本文件读取）；
  - 仍生成预览图（使用默认视角渲染两个球）。

## 开发步骤（实施清单）
1) 配置：为 `--interact`、`--snap_to_surface`、`--ui_port` 等添加解析；
2) 交互模块：实现 `utils/interactive_select.py`（Dash），封装渲染、事件与返回；
3) 主流程：`main.py` 调用交互并将结果写入 `initialization`，保存 JSON/快照；
4) 初始化：在 `initialize_dual_hand_at_targets` 中按“物体中心”为朝向对齐，按 φ_k 规则扭转（左右共享 φ_k）；
5) 记录：在 `Logger` 或额外日志中记录用户选点；
6) 文档：在 `dev/scripts.md` 增加交互指令示例与注意事项。

## 运行示例（交互）
```bash
python BimanGrasp-Optimization/main.py \
  --name interact_demo \
  --object_code merged_collision_300k_wt \
  --batch_size 16 --num_iterations 1000 \
  --interact --target_distance 0.25 --vis --vis_frame_stride 50
```
- 页面确认 A、B 后自动开始优化；结果与交互信息保存在 `data/experiments/interact_demo/results/`。

