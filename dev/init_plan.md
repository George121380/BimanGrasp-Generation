# 手部初始化机制概览（BimanGrasp-Generation）

## 入口与调用流程
- 在优化开始前，`main.py` 会：
  - 构建右手 `HandModel`；
  - 构建并 `initialize` 对象 `ObjectModel`（根据 `object_code_list` 与 `batch_size_each` 准备批量对象尺度与点云采样）；
  - 调用 `initialize_dual_hand(right_hand_model, object_model, args)` 同时生成左/右手初始位姿与接触点；
  - 用生成的左右手实例构造 `BimanualPair`，进入能量计算与优化循环。

## 左手初始化（位姿）
- 凸包与采样：对每个对象的网格取凸包并“外扩”，在其表面均匀选取 `batch_per_obj` 个点 `p`，并用原始凸包的最近点方向得到法向 `n`。
- 随机参数：
  - 距离 `distance ∈ [distance_lower, distance_upper]`；
  - 角度抖动（锥角）`theta ∈ [theta_lower, theta_upper]`；
  - 方位 `azimuth ∈ [0, 2π)`；滚转 `roll ∈ [0, 2π)`。
- 姿态求解：构造世界旋转 `world_rot` 与抖动旋转 `cone_rot`，再与固定手姿 `hand_rot` 组合：
  - 平移：\( translation = p - distance · (world\_rot · cone\_rot · \hat{z}) \)
  - 旋转（左手）：\( rotation = world\_rot · cone\_rot · (-hand\_rot) \)
- 手部状态编码：最终 `hand_pose = [translation(3), rotation(3×3)的前两列(6), joints]`（在代码中通过 `rotation.transpose(1, 2)[:, :2]` 展平得到 6D 旋转表示）。

## 右手初始化（镜像）
- 镜像法向与点：对左手的 `n, p` 在 x、y 轴取相反数得到右侧放置基准；然后独立随机一组 `distance/angles`。
- 姿态求解（右手）：
  - 平移与左手同式：\( translation\_right = p - distance · (world\_rot · cone\_rot · \hat{z}) \)
  - 旋转（右手）：\( rotation\_right = world\_rot · cone\_rot · hand\_rot \)（注意没有负号）。
- 同样将右手的平移、旋转前两列与关节角拼接为 `hand_pose`。

## 关节与接触点初始化
- 关节角：
  - 左手围绕手工均值 `LEFT_HAND_JOINT_MU`、右手围绕 `RIGHT_HAND_JOINT_MU`；
  - 标准差按关节上下界跨度与 `jitter_strength` 确定，使用“截断正态”采样并裁剪在关节合法范围内。
- 接触点：为每只手随机采样 `num_contacts` 个接触候选的全局索引，随后经正运动学变换到全局坐标，缓存至 `contact_points`。

## 批处理与索引关系
- 设对象数 `N_obj = len(object_code_list)`，每对象样本数 `batch_per_obj = model.batch_size`：
  - 总 batch：`total_batch_size = N_obj * batch_per_obj`；
  - 第 `i` 个对象（0-based）的第 `j` 个样本的“全局样本索引”为 `global_idx = i * batch_per_obj + j`；
  - 该索引在初始化与可视化/记录中均用于选择具体样本实例。

## 主要可配置项（影响初始化的分布）
- `InitializationConfig`（`utils/config.py`）：
  - `distance_lower / distance_upper`：与物体的初始距离范围；
  - `theta_lower / theta_upper`：朝向抖动范围（锥角）；
  - `jitter_strength`：关节角截断正态的随机强度；
  - `num_contacts`：每只手的初始接触点数量。
- `ModelConfig.batch_size`：每个对象生成的样本数量（影响 `batch_per_obj`）。
- `ExperimentConfig.object_code_list`：参与初始化的对象集合。

## 初始化产物（进入优化前的状态）
- 对每个样本，左右手均已具备：
  - `global_translation`、`global_rotation`（由 6D 旋转恢复）；
  - `hand_pose`（含平移、6D 旋转、关节角）；
  - `current_status`（正运动学结果，各链路位姿）；
  - `contact_points`（根据随机接触索引、经正运动学与全局变换得到的接触点坐标）。
- 这些状态将进入能量计算，作为 MALA 提案与 MH 接受的起点。

## 定点初始化（A/B）功能实现计划
- 目标：用户指定物体表面两个点 A、B（在物体坐标系下，或世界坐标系下若物体位姿为单位），左手初始化在 A 点附近并且手掌朝向 A，右手初始化在 B 点附近并且手掌朝向 B。

- 接口设计（新增 CLI/配置）：
  - `--init_at_targets`（bool，默认 false）：是否启用定点初始化。
  - `--left_target "x y z"`：左手目标点 A（浮点，物体坐标系）。
  - `--right_target "x y z"`：右手目标点 B（浮点，物体坐标系）。
  - `--target_distance`（float，默认沿法向/指向方向后退的距离，例如 0.22）。
  - `--target_jitter_dist`（float，默认 0.0）：距离抖动范围（±），可选。
  - `--target_jitter_angle`（float，默认 0.0，弧度）：朝向抖动半角（绕目标方向锥抖动），可选。
  - 若用户只提供 A、不提供 B，则右手可选：
    - 镜像 A（与现方案一致，x、y 取反）得到 B；或
    - 使用同 A（双手同点，适合对向抓取）。

- 主要步骤（替代/分支于 `initialize_dual_hand`）：
  1) 解析 A/B：
     - 使用 `ObjectModel` 的缩放（`object_scale_tensor`）将 A/B 从物体坐标系缩放到实际尺寸：`A_scaled = scale * A`，`B_scaled = scale * B`（按当前样本的尺度）。
     - 获取 A/B 附近的法向：
       - 简便做法：用 `trimesh` 最近点法线或 `ObjectModel.calculate_distance(..., with_closest_points=True)` 在密集点云上近似获得法向；
       - 或直接使用从手到点的指向向量作为朝向基准（无需法向）。
  2) 计算手位姿：
     - 方向设定：设手掌局部 +z 为抓取朝向。令 `dir_A = normalize(A_world - hand_pos)`，我们将先设 `world_rot_A` 使得 +z 轴对齐 `dir_A`；同理得到 `world_rot_B`。
     - 平移：`translation_left = A_world - d * dir_A`，`translation_right = B_world - d * dir_B`，其中 `d = target_distance + U(-target_jitter_dist, target_jitter_dist)`。
     - 朝向抖动：在 `dir_A`、`dir_B` 周围半角 `target_jitter_angle` 的圆锥内采样一个方向，更新 `world_rot_*`。
     - 左右手差异：沿用当前实现的微调旋转（`hand_rot`）：
       - 左手：`rotation_left = world_rot_A @ (-hand_rot)`；
       - 右手：`rotation_right = world_rot_B @ hand_rot`。
     - 拼装 `hand_pose`：平移 + 6D 旋转前两列 + 关节角（关节角仍按截断正态采样）。
  3) 接触点初始化（可选强化）：
     - 可在各手的 `contact_candidates` 中，选择距离目标点最近的若干个候选作为初始 `contact_point_indices`（而非完全随机），提高起始可行性；
     - 仍保留随机成分以避免过拟合（例如：从最近的 K 个中均匀随机采样）。
  4) 批处理：
     - 若 `batch_per_obj > 1`，默认将相同的 A/B 复制到该对象的每个样本；或支持 `--left_targets_file / --right_targets_file` 传入一批点（与 batch 对齐）。

- 代码改动点：
  - `utils/config.py`：
    - 新增 `InitializationConfig` 子项或新建 `TargetInitConfig`，含上述 CLI 参数；`ExperimentConfig.update_from_args` 解析对应字段。
  - `utils/initializations.py`：
    - 新增函数 `initialize_dual_hand_at_targets(right_hand_model, object_model, args)`；
    - 内部实现上述 A/B 定点位姿求解与关节、接触点初始化；
    - 保持与现有 `set_parameters`、`hand_pose` 格式一致。
  - `BimanGrasp-Optimization/main.py`：
    - 在 `setup_models()` 中根据 `args.init_at_targets` 分支调用 `initialize_dual_hand_at_targets(...)` 或原始 `initialize_dual_hand(...)`。

- 伪代码示例：
```
if args.init_at_targets:
    left_targets = parse_vec3(args.left_target)    # (3,) or (batch,3)
    right_targets = parse_vec3(args.right_target)  # (3,) or (batch,3)
    left_hand_model, right_hand_model = initialize_dual_hand_at_targets(
        right_hand_model, object_model, args,
        left_targets, right_targets
    )
else:
    left_hand_model, right_hand_model = initialize_dual_hand(
        right_hand_model, object_model, args
    )
```

- 注意事项：
  - 坐标系需与 `ObjectModel` 一致（物体世界位姿为单位时，物体坐标即世界坐标；若未来引入物体位姿，需做变换）。
  - 多对象批处理时，A/B 应分别对应对象索引；或限制启用定点初始化时只处理单对象。
  - 与可视化、指标记录的 `global_idx` 兼容：定点初始化完成后可使用同样的样本索引进行渲染与记录。
