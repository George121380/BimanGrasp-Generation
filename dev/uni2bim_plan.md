标题：在不破坏默认行为的前提下新增 uni2bim 模式的最小实现计划

目标回顾
- 在不重构整体框架的前提下，增设 `uni2bim` 模式，使左右手的优化完全解耦：
  - 关闭所有跨手耦合项（互穿、联合稳定性、任何显式“联合 force closure”）。
  - 仅保留每只手各自对物体/自身的能量项，并在总能量里相加。
  - 非 `uni2bim` 模式下行为与当前版本完全一致。

代码现状（关键定位）
- 总能量汇总在：`BimanGrasp-Optimization/utils/bimanual_energy.py` 的 `BimanualEnergyComputer.compute_all_energies`

```276:335:BimanGrasp-Optimization/utils/bimanual_energy.py
    def compute_all_energies(self, bimanual_pair: BimanualPair, object_model, 
                           verbose: bool = False) -> EnergyTerms:
        ...
        energy_dis = self.contact_distance_computer.compute(bimanual_pair, object_model)
        energy_pen = self.penetration_computer.compute_object_penetration(bimanual_pair, object_model)
        energy_spen = self.penetration_computer.compute_self_penetration(bimanual_pair)
        energy_joints = bimanual_pair.compute_joint_limits_energy()
        ...
        # FC & VEW 当前用合并的接触点构造一次 G
        ...
        energy_fc, energy_vew = self.grasp_matrix_computer.compute_fc_and_vew(G)
        ...
        energy_total = (energy_fc + 
                  self.config.w_dis * energy_dis + 
                  self.config.w_pen * energy_pen + 
                  self.config.w_spen * energy_spen + 
                  self.config.w_joints * energy_joints + 
                  self.config.w_vew * energy_vew)
```

- 自穿/互穿项在：`PenetrationComputer.compute_self_penetration`，其返回包含“单手自穿 + 跨手互穿”三部分之和。

```182:201:BimanGrasp-Optimization/utils/bimanual_energy.py
    def compute_self_penetration(self, bimanual_pair: BimanualPair) -> torch.Tensor:
        # 单手自穿
        left_spen, right_spen = bimanual_pair.apply_to_both(lambda h: h.self_penetration())
        # 跨手互穿（需要在 uni2bim 关闭）
        surface_points_right = bimanual_pair.right.surface_point.detach().clone()
        inter_pen = bimanual_pair.left.cal_distance(surface_points_right)
        inter_pen = torch.clamp(inter_pen, min=0)
        return left_spen + right_spen + inter_pen.sum(-1)
```

- 单手自穿实现（可直接复用）：`HandModel.self_penetration`

```322:346:BimanGrasp-Optimization/utils/hand_model.py
    def self_penetration(self):
        ...
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1,2))
```

- 接触距离 `E_dis` 已按手分别对物体求和：`ContactDistanceComputer.compute`（无需修改）。

```227:233:BimanGrasp-Optimization/utils/bimanual_energy.py
        distance_left, _ = object_model.cal_distance(bimanual_pair.left.contact_points)
        distance_right, _ = object_model.cal_distance(bimanual_pair.right.contact_points)
        energy_dis = torch.sum(distance_left.abs(), dim=-1) + torch.sum(distance_right.abs(), dim=-1)
```

- 入口与配置：
  - CLI 在 `BimanGrasp-Optimization/main.py`（需要加 `--mode`）。
  - 配置在 `BimanGrasp-Optimization/utils/config.py` 的 `ExperimentConfig`/`EnergyConfig`（建议在 `EnergyConfig` 增加布尔开关以便下传）。

实现步骤（严格最小改动）
1) 增加模式开关（默认不启用，保持现有行为）
   - 在 `EnergyConfig` 新增字段：`use_uni2bim: bool = False`。
   - 在 `main.py` 的 argparse 增加参数：`--mode`，取值 `default|uni2bim`（默认 `default`）。
   - 在 `ExperimentConfig.update_from_args` 中：
     - 读取 `args.mode`；若为 `uni2bim`，则设 `config.energy.use_uni2bim = True`。
     - 其余不变，保持向后兼容。

2) 关闭跨手耦合项（仅在 uni2bim 生效）
   - 互穿项（inter-hand penetration / `E_bimpen` 等价）：
     - 不修改 `compute_self_penetration` 函数本身。
     - 在 `BimanualEnergyComputer.compute_all_energies` 内，若 `use_uni2bim=True`，直接用：
       `left_spen, right_spen = bimanual_pair.apply_to_both(lambda h: h.self_penetration())`，
       并令 `energy_spen = left_spen + right_spen`，而不调用 `penetration_computer.compute_self_penetration`。
   - 协同稳定性项（`E_vew` 或任何依赖双手合并 G 的 VEW/全局稳定项）：
     - 不修改 `WrenchVolumeComputer` 实现。
     - 若 `use_uni2bim=True`，令 `energy_vew = torch.zeros(batch_size, device=self.device)`（等价于权重=0）。
   - 若存在显式双手联合 FC 项（当前实现中 FC 默认合并两手接触点）：在 uni2bim 下禁用该合并方式（见第3步拆分）。

3) 唯一结构性修改：将 FC 拆分为左右手两份并相加（仅在 uni2bim 生效）
   - 在 `BimanualEnergyComputer.compute_all_energies` 中添加分支：
     - 若 `use_uni2bim=True`：
       1) 仅基于左手接触点/法向构造 `G_left`，用 `compute_fc_and_vew(G_left)` 得到 `energy_fc_left`（忽略其 vew）。
       2) 仅基于右手接触点/法向构造 `G_right`，得到 `energy_fc_right`。
       3) 令 `energy_fc = energy_fc_left + energy_fc_right`。
     - 若 `use_uni2bim=False`：保持当前“合并左右手接触点一次构造 G”的逻辑不变。

4) 其它能量项按手独立相加（无需新造函数）
   - `E_dis`：当前已分别对左右手求和，无需变动。
   - `E_objpen`：`compute_object_penetration` 对左右手加和，无需变动。
   - `E_selfpen`：见第2步；在 uni2bim 下仅保留单手内部自穿和。
   - `E_joint`：`BimanualPair.compute_joint_limits_energy` 已分别计算并求和，无需变动。

5) 总能量在 uni2bim 下的表达式
   - 仅在 `compute_all_energies` 的加权汇总处，按条件分支使用上述的 `energy_fc`、`energy_spen`、`energy_vew`：
     - `E_total_uni2bim = (E_fc_left + E_fc_right) + w_dis*E_dis + w_pen*E_objpen + w_spen*(E_spen_left + E_spen_right) + w_joints*E_joint + 0*E_vew`
   - 默认模式保持原有：`E_total = E_fc(左右手合并G) + w_dis*E_dis + w_pen*E_objpen + w_spen*(自穿+互穿) + w_joints*E_joint + w_vew*E_vew`。

文件级改动清单（精确到函数/位置）
1) `BimanGrasp-Optimization/utils/config.py`
   - dataclass `EnergyConfig`：新增字段 `use_uni2bim: bool = False`。
   - `ExperimentConfig.update_from_args`：读取 `args.mode` 并设置 `self.energy.use_uni2bim`（仅当 `mode == 'uni2bim'`）。

2) `BimanGrasp-Optimization/main.py`
   - argparse 新增：`parser.add_argument('--mode', default='default', choices=['default','uni2bim'])`。
   - 其余保持，用现有 `create_config_from_args(args)` 将 `mode` 传入并在 `update_from_args` 中落地到 `EnergyConfig.use_uni2bim`。

3) `BimanGrasp-Optimization/utils/bimanual_energy.py`
   - 在 `BimanualEnergyComputer.compute_all_energies` 内部：
     - 引入分支 `if self.config.use_uni2bim:`：
       - `E_selfpen`：通过 `bimanual_pair.apply_to_both(lambda h: h.self_penetration())` 计算左右手自穿并相加。
       - `E_fc`：为左/右手分别构造 G，分别调用 `compute_fc_and_vew(G_side)`，取各自的 FC 值并相加。
       - `E_vew`：置零张量（保持尺寸与设备一致）。
     - `else`：维持现有合并逻辑（完全不变）。
   - 注意：不修改 `ForceClosureComputer`、`WrenchVolumeComputer`、`PenetrationComputer` 的函数签名与实现，只在汇总处选择性绕过/置零。

边界与兼容性说明
- 默认运行（无 `--mode` 或 `--mode default`）完全走旧路径：无任何行为差异。
- 若单侧缺失接触点：
  - 默认模式已有回退；
  - uni2bim 模式下，FC 将只计算存在接触点的一侧，另一侧贡献为零（与“按手独立 + 求和”一致）。
- 计算开销：uni2bim 将对左右手各做一次 SVD（以当前实现为准），属可接受范围；不做额外重构优化。

验证计划（强制）
1) 回归一致性（默认模式）
   - 固定随机种子与输入，运行前后对比第一步迭代的 `energy_total`、各分量日志（允许浮点微差，但分支/路径应一致）。
2) uni2bim 梯度来源独立性
   - 随机小批测试：仅对左手参数保留梯度，右手参数 `requires_grad=False`；检查反向传播：右手梯度恒为零；交换左右同理。
3) 项级别数值检查
   - 互穿项：uni2bim 下 `w_spen*E_spen` 不应包含跨手距离；
   - VEW：uni2bim 下 `w_vew*E_vew` 的日志应恒为 0；
   - FC：uni2bim 下 `E_fc` 等于左右 FC 之和（可打印两侧 FC 以断言）。

CLI 使用示例
- 默认（不变）：
  - `python BimanGrasp-Optimization/main.py --name exp_default`
- 启用 uni2bim：
  - `python BimanGrasp-Optimization/main.py --name exp_u2b --mode uni2bim`
- 大规模脚本 `tools/scale_runner.py`：通过 `--main_extra_args -- --mode uni2bim` 透传给 `main.py`。

风险与回滚
- 风险：`compute_all_energies` 分支改动引入维度/设备不一致；对策：严格用 `batch_size`、`device=self.device` 初始化零张量，沿用现有代码风格。
- 回滚：单文件改动，移除 `use_uni2bim` 分支与 CLI 参数即可完全回退。

实施顺序与工时（建议）
1) 配置与 CLI（30min）：`EnergyConfig` 增加字段、`main.py` 增加 `--mode`、`update_from_args` 落地。
2) 汇总分支（60min）：在 `compute_all_energies` 实现 uni2bim 分支（FC 拆分、自穿改为按手、VEW 置零）。
3) 快速验证（30min）：单/双侧接触点存在与否、日志检查、简单梯度检查。
4) 扩展验证（可选，60min）：小规模迭代、比对默认与 uni2bim 曲线差异。

备注
- 所有新增/修改代码内注释一律使用英文；
- 临时文件（如调试日志、临时可视化）请在验证后删除，避免污染仓库。

