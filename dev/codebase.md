# BimanGrasp-Generation 代码库总览

## 仓库结构
- `README.md`: 项目简介、安装步骤、用法、演示图、引用。
- `install.sh`: 自动化 conda 环境引导（PyTorch 2.1.0 CUDA 11.8、PyTorch3D、TorchSDF、pytorch_kinematics）。
- `assets/figs`: 五个演示对象的抓取渲染结果。
- `data/meshdata`: 演示对象的预处理网格（COACD 分解）。
- `data/experiments`: 默认实验输出（`logs/`、`results/config.txt`）；脚本会为每次运行清理/创建子目录。
- `BimanGrasp-Optimization/`: 生成双手抓取的核心优化代码。
- `thirdparty/`: 版本化依赖 `TorchSDF`（有符号距离运算）与 `pytorch_kinematics`（MJCF 解析、运动学）。

## 核心实验流水线（`BimanGrasp-Optimization/`）
- `main.py`: 入口，定义 `GraspExperiment`。
  - 设置设备/随机种子（`setup_environment`）。
  - 加载右手 `HandModel`，初始化 `ObjectModel`，通过 `initialize_dual_hand` 镜像采样左手，打包为 `BimanualPair`（`setup_models`）。
  - 实例化 `BimanualEnergyComputer` 与 `MALAOptimizer`（`setup_optimization`）。
  - 通过 `ensure_directory` 创建日志/结果目录（`setup_logging`），写入运行配置，构建用于 TensorBoard 指标的 `Logger`。
  - `run_optimization`:
    * 计算初始能量并反向传播以填充梯度。
    * 按 `num_iterations` 进行 Langevin 提案（`MALAOptimizer.langevin_proposal`）与 Metropolis-Hastings 接受步骤。
    * 更新被接收样本的能量缓存，每步记录汇总。
  - `save_final_results`: 通过 `save_grasp_results` 导出每对象一个 `.npy` 的抓取字典。
  - 可选分析钩子（`cProfile`、`memory_profiler`）。
- `visualization.py`: 载入保存的抓取，重建手/物体网格，渲染交互式 Plotly；期望 `results/*.npy` 文件。

## 关键工具模块（`utils/`）
- `config.py`: 数据类层级（`ExperimentConfig`）及其子配置 `PathConfig`、`EnergyConfig`、`OptimizerConfig`、`InitializationConfig`、`ModelConfig`；通过 `create_config_from_args` 与 CLI 集成，并提供如 `total_batch_size` 等派生属性。
- `hand_model.py`: 基于 MJCF 构建运动链，加载网格/接触/穿透关键点，采样表面点，维护批量手部姿态状态，计算有符号距离/自穿透，变换接触候选点，提供 Plotly 导出辅助。
- `object_model.py`: 加载 COACD 网格，采样最远点表面，管理按对象的随机缩放，使用 TorchSDF 计算物体 SDF 距离/法向。
- `initializations.py`: 在膨胀的物体凸包周围生成左右手姿态；用截断正态初始化关节，随机采样接近方向。
- `bimanual_handler.py`: `HandState`、`EnergyTerms`、`GraspData` 等结构；`BimanualPair` 协调整双手操作（状态保存/回滚、关节限制/自穿透/手-物体穿透计算）；序列化辅助（`hand_pose_to_dict`、`save_grasp_results`）。
- `bimanual_optimizer.py`: MALA（Metropolis-Adjusted Langevin）+ RMSProp 预条件，温度调度，自适应接触重采样，拒绝提案的状态回滚。
- `bimanual_energy.py`: 通过共享一次 SVD（`GraspMatrixComputer`）组合计算接触距离、物体穿透、自穿透、关节限制、力闭合与扳手椭球体积。
- `common.py`: 设备设置、可复现性、张量运算、日志（`Logger` 写入 TensorBoard 标量）、旋转转换与采样等工具函数。

## 数据流与输出
1. 配置选择对象代码（`data/meshdata/<object>/coacd/decomposed.obj`）与批大小。
2. 初始化生成镜像的手部姿态/接触候选；`HandModel.set_parameters` 缓存正运动学结果与接触点变换。
3. `BimanualEnergyComputer` 评估加权能量；梯度通过手部姿态传播，用于 MALA 更新。
4. 接受的提案被记录；最终每个对象一个 `.npy`，在 `data/experiments/<name>/results` 内保存 `GraspData` 字典（尺度、`qpos_left/right`、起始姿态、各能量项）。
5. TensorBoard 日志位于 `data/experiments/<name>/logs`。

## 第三方集成
- `TorchSDF`: 网格 SDF 查询的自定义 CUDA 运算（用于 `HandModel.calculate_distance` 与 `ObjectModel.calculate_distance`）。
- `pytorch_kinematics`: 解析 MJCF，为 ShadowHand 提供正运动学。
- 外部 Python 依赖：PyTorch 2.1、PyTorch3D、transforms3d、trimesh、plotly、scipy、tensorboard 等。安装脚本会编译 TorchSDF 扩展并以可编辑方式安装运动学库。

## 运行说明
- 默认命令行（摘自 `README.md`）：
  ```bash
  conda activate bimangrasp
  python BimanGrasp-Optimization/main.py --name test
  python BimanGrasp-Optimization/visualization.py --object_code <object> --num <index>
  ```
- 建议使用 GPU；设备选择由 `--gpu`/`ExperimentConfig.gpu` 控制（设置 `CUDA_VISIBLE_DEVICES`）。
- `ExperimentConfig` 中的目录相对于 `BimanGrasp-Optimization/`；在其他工作目录运行时需确保路径有效。
- 演示对象仅包含五个样例网格；新增资产需遵循 DexGraspNet 的处理流程。

## 维护注意事项
- 长时间运行（默认 10k 次迭代）可用内置 `cProfile` 进行分析；除非手动调用 `save_intermediate_results`，否则仅在最终步骤保存结果。
- 手/物体 SDF 计算假定有 CUDA；若需 CPU 回退，可能需要修改 `HandModel` 中的 `.cuda()` 调用。
- TorchSDF 二进制 `_C.so` 为 Python 3.8 预编译；环境不同需重新编译。
- 记录会清理已有实验目录；自定义运行请使用唯一的 `--name` 以避免误删。
