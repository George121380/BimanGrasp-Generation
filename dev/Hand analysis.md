### 背景
本仓库当前的可视化与优化流程默认使用 ShadowHand（MJCF，`mjcf/left_shadow_hand.xml`、`mjcf/right_shadow_hand.xml`）。现在新增了 Inspire Hand OY（简称 PSI‑OY，URDF，位于 `BimanGrasp-Optimization/psi-oy/`），需要在保持原有 ShadowHand 支持的同时，扩展 `visualization.py` 以支持新手型的结果可视化。

本文档对两套手型在文件格式、关节/DOF、命名与控制方式等方面进行梳理，并给出最小侵入、可扩展的改造方案与注意事项。

---

### 资产与文件格式对比
- **ShadowHand（现有）**
  - **格式**: MJCF（MuJoCo XML）
  - **路径**: `BimanGrasp-Optimization/mjcf/`
  - **几何**: `mjcf/meshes/` 下 `.obj`，在 MJCF `<mesh>` 中通过 `name="robot0:*"` 引用
  - **接触/穿透关键点**: `left_hand_contact_points.json`、`right_hand_contact_points.json`、`penetration_points.json`
  - **HandModel 构建**: 依赖 `pytorch_kinematics.build_chain_from_mjcf`

- **PSI‑OY（新增）**
  - **格式**: URDF
  - **路径**: `BimanGrasp-Optimization/psi-oy/InspireHand_OY_Left/InspireHand_OY_Left.urdf` 与 `.../Right/...Right.urdf`
  - **几何**: URDF `<mesh filename="meshes/*.obj"/>`，文件相对 URDF 目录给出
  - **接触/穿透关键点**: 暂无（仅影响优化/能量项；纯可视化不依赖）
  - **HandModel 构建（预期）**: 依赖 `pytorch_kinematics.build_chain_from_urdf`

结论：两者在“运动学描述格式”和“几何引用方式”上不同；PSI‑OY 使用 URDF 的相对路径，ShadowHand 使用 MJCF 的命名引用。`HandModel` 需要统一在内部对 MJCF/URDF 的解析与网格加载逻辑做分支处理。

---

### 关节/DOF 与命名对比
- **ShadowHand（MJCF）**
  - 关节命名：`robot0:FFJ3, FFJ2, FFJ1, FFJ0, ... , LFJ4, ..., THJ4..THJ0` 共 22 个 DOF（含无名腕部自由度在外部以位姿表示，非关节）
  - 现有代码中 `utils/config.py` 定义了固定的 `JOINT_NAMES` 列表（22 个），并且 `visualization.py` 也写死了同样次序
  - 保存结果时（`utils/bimanual_handler.py::hand_pose_to_dict`）使用上述全局 `JOINT_NAMES` 进行字典化

- **PSI‑OY（URDF）**
  - 关节命名（左手示例）：`hand1_joint_link_1_1`、`hand1_joint_link_1_2`、`hand1_joint_link_1_3(mimic)`、`..._2_1`、`..._2_2(mimic)`、`..._3_1`、`..._3_2(mimic)`、`..._4_1`、`..._4_2(mimic)`、`..._5_1`、`..._5_2(mimic)`；若按“非 fixed”统计共有 10 个 revolute 关节，其中 4 组存在 mimic 关系
  - 独立 DOF 估计：6（拇指 2，自由指各 1，合计 2+4），但链上会包含 mimic 关节节点（需要以 `pytorch_kinematics` 的实际实现为准：有的实现会将 mimic 视作从属，不计入自由度输入向量；有的会保留节点但内部用从属约束计算）
  - 右手命名与左手相似但前缀为 `hand2_joint_*`，轴向定义也有差异（URDF 通过 `axis xyz=` 与局部坐标系体现）

结论：两者 DOF 数量与命名体系均不同；PSI‑OY 含 mimic 关节，需明确输入向量是否只提供“独立 DOF”。因此，任何“硬编码的关节名列表”和“固定长度向量拼接”都需要重构为“按模型/链动态获取”。

---

### 位姿/字典键格式对比（可视化与保存）
- 现有保存/可视化统一使用：
  - 平移键：`WRJTx, WRJTy, WRJTz`
  - 旋转键：`WRJRx, WRJRy, WRJRz`（欧拉角）
  - 关节键：由 `JOINT_NAMES` 提供
  - 在 `visualization.py` 中，欧拉角经 `transforms3d.euler2mat` 转矩阵，再取前两列展平为 Rot6D（`HandModel` 期望 `3+6+ndofs`）

建议：继续复用统一的平移/旋转键名（不随手型变化），仅将“关节键名列表”改为按手模型动态获取，以兼容多手型并避免写死长度与次序。

---

### HandModel 适配 URDF 的必要改造
现状：`utils/hand_model.py` 在构造函数里写死了 `build_chain_from_mjcf`，并且在加载 mesh 时假定 MJCF 的 `visual.geom_param[0]` 是 `"robot0:XXX"`，通过 `split(":")[1] + ".obj"` 在 `mesh_path` 下寻址。

改造要点：
- 解析层：
  - 根据文件后缀自动选择 `build_chain_from_mjcf` 或 `build_chain_from_urdf`
  - 新增可选参数 `model_format`（`"auto"|"mjcf"|"urdf"`），默认 `auto`

- 几何层：
  - 当 `visual.geom_type == "mesh"`：
    - 若是 MJCF：保持现状（`robot0:*` 名称 -> `mesh_path/NAME.obj`）
    - 若是 URDF：`visual.geom_param[0]` 已是相对路径（如 `meshes/hand1_link_1_1_visuals.obj`），应当使用 `os.path.join(mesh_path, visual.geom_param[0])` 直接加载；同时识别 `visual.geom_param[1]` 作为 scale（若存在）

- 关节上下界：
  - 目前左手会将上下界取负（镜像）以对称到右手；URDF 左右手分别提供了自身的范围与轴定义，建议：
    - 对 MJCF 维持现有 mirror 行为
    - 对 URDF 默认不镜像（或通过新布尔开关 `mirror_limits` 控制，URDF 下默认 False）

- 接触/穿透点：
  - 可视化不要求；优化若要启用 PSI‑OY，需要补充与链接名匹配的 JSON（键为链路名，值为 Nx3 点坐标）

---

### visualization.py 适配方案（最小侵入）
目标：不改动结果文件格式（仍用统一的 `WRJT*`、`WRJR*` 键），在加载时根据“手模型的真实 DOF 名单”动态拼装手部姿态向量。

具体方案：
1) 新增 CLI 参数：`--hand_type {shadowhand, psi_oy}` 或更通用的 `--left_hand_file`/`--right_hand_file` 与 `--format {mjcf, urdf}`

2) 根据参数初始化对应的 `HandModel`：
   - ShadowHand：
     - `mjcf_path='mjcf/left_shadow_hand.xml'` / `...right...`
     - `mesh_path='mjcf/meshes'`
     - `contact_points_path/penetration_points_path`：沿用现有 JSON
   - PSI‑OY：
     - `urdf_path='psi-oy/InspireHand_OY_Left/InspireHand_OY_Left.urdf'`（或 Right）
     - `mesh_path='psi-oy/InspireHand_OY_Left'`（或 Right），使得 `mesh_path + filename` 能解析到 `meshes/*.obj`
     - `contact_points_path=None`、`penetration_points_path=None`

3) 从结果字典恢复位姿：
   - 统一读取平移与欧拉角：`[qpos[k] for k in TRANSLATION_NAMES + ROTATION_NAMES]`
   - 将欧拉角转 Rot6D（与现实现一致）
   - 关键：关节角应按 `hand_model.joints_names` 的顺序映射：
     - 对每个 `name in hand_model.joints_names`，若 `name` 存在于 `qpos`：取其值；否则（例如含 mimic 或未保存的派生关节）置 0 或按需要映射/派生
   - 拼为 `torch.tensor([Tx,Ty,Tz] + Rot6D + joint_list)`，形状 `1 x (3+6+n_dofs)`

4) 组装图层并渲染（沿用现有 Plotly 流程）：
   - `object_model` 与配色、背景不变
   - `with_contact_points=False` 作为默认（PSI‑OY 无接触点 JSON）

示例（伪代码，仅示意）：
```python
rot = euler2mat(qpos['WRJRx'], qpos['WRJRy'], qpos['WRJRz'])
rot6d = rot[:, :2].T.reshape(-1).tolist()
joint_vals = [qpos.get(name, 0.0) for name in hand_model.joints_names]
hand_pose = torch.tensor([qpos['WRJTx'], qpos['WRJTy'], qpos['WRJTz']] + rot6d + joint_vals, dtype=torch.float, device=device).unsqueeze(0)
hand_model.set_parameters(hand_pose)
```

---

### 结果文件兼容性与保存逻辑建议
- 现有保存函数 `hand_pose_to_dict` 使用全局 `JOINT_NAMES`（ShadowHand 固定 22 项）；当切换至 PSI‑OY 时会与 `n_dofs` 不匹配。
- 建议：
  - 将 `hand_pose_to_dict` 改为基于模型的关节名列表，而非全局常量：
    - 方案 A：修改签名为 `hand_pose_to_dict(hand_pose, joint_names, translation_names, rotation_names)`
    - 方案 B：将其移动到 `HandModel` 或通过传入 `hand_model` 实例以读取 `hand_model.joints_names`
  - `utils/config.py` 中的 `JOINT_NAMES` 仅保留 ShadowHand 的默认值；新增 `get_joint_names_from_model(hand_model)` 的通用路径
- 平移/旋转键仍沿用 `WRJT* / WRJR*`，保证跨手型一致性与可视化脚本的稳定性。

---

### 注意事项（坑点汇总）
- **URDF mimic 关节**：明确 `pytorch_kinematics.build_chain_from_urdf` 对 mimic 的处理方式；若输入 DOF 仅需“独立关节”，需从 `chain.get_joint_parameter_names()` 验证长度，并据此构造 `hand_pose` 尾部；若包含 mimic，需要确保结果中也保存了对应键，或在载入时自动派生值。
- **左/右手关节上下界镜像**：ShadowHand 目前对左手做镜像处理；URDF 左右手已分离，建议默认不镜像，或通过显式开关控制。
- **网格路径**：URDF 的 `<mesh filename="meshes/...obj">` 是“相对 URDF 文件夹”的路径；需正确设置 `mesh_path` 为 URDF 所在目录（而非单纯 `meshes/`）。
- **接触点 JSON**：PSI‑OY 若要参与优化与能量项，需要新增与其链路名一致的 JSON。仅做可视化可传 `None` 并关闭 `with_contact_points`。
- **结果兼容**：在既有 `.npy` 结果文件中，关节键为 ShadowHand 的名；对 PSI‑OY 新结果，务必以“模型实际的关节名”保存（按上文保存逻辑调整后自动满足）。`visualization.py` 将按模型自适应读取。

---

### 最小改造清单（实施顺序）
1) 改造 `utils/hand_model.py`
   - 自动识别 `mjcf|urdf`，分别用 `build_chain_from_mjcf|build_chain_from_urdf`
   - `mesh` 加载分支：MJCF 走 `robot0:*` 名称，URDF 走 `mesh_path + filename`
   - 新增可选参数 `model_format='auto'`、`mirror_limits`（URDF 默认 False，MJCF 维持现状）

2) 改造 `visualization.py`
   - 新增 `--hand_type` 或 `--left/right_*` 文件路径与 `--format` 参数
   - 初始化 `HandModel` 时传入对应路径/格式；PSI‑OY 关闭接触点
   - 使用 `hand_model.joints_names` 动态从 `qpos` 映射关节角，避免硬编码 `joint_names`

3)（建议）改造保存逻辑 `utils/bimanual_handler.py`
   - 将 `hand_pose_to_dict` 从全局 `JOINT_NAMES` 改为参数化或模型驱动
   - 保持 `WRJT* / WRJR*` 不变

4)（后续可选）为 PSI‑OY 标注 `contact_points.json` 与 `penetration_points.json`，解锁优化与能量评估

---

### 验证与测试建议
- 单元检查：
  - 实例化 PSI‑OY 左/右手模型（URDF），调用 `get_plotly_data(i=0)` 是否能正确渲染
  - 打印 `hand_model.n_dofs`、`hand_model.joints_names` 并与 URDF 关节列表比对
  - 将随机 `hand_pose`（角度在关节上下界内）设置后，检查 Plotly 是否更新且无异常

- 结果回放：
  - 使用 ShadowHand 的历史 `.npy` 进行可视化，确认不受影响
  - 使用 PSI‑OY 的新 `.npy`（按新保存逻辑写出）回放，确认姿态一致

- 边界：
  - 对缺失的关节键（如 mimic 未显式保存）采用 0 填充或按规则派生；记录日志提示

---

### 里程碑
- M1：完成 `HandModel` 对 URDF 的兼容与 `visualization.py` 动态关节映射（仅可视化，无接触点）
- M2：完成 `hand_pose_to_dict` 改造，统一“按模型导出”
- M3：为 PSI‑OY 标注接触/穿透点 JSON，打通优化与能量评价






