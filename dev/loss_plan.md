# 优化指标记录方案（BimanGrasp-Generation）

## 目标
- 在优化过程中记录关键指标（如 loss/energy 及其各项子能量、接受率、温度、步长等），便于离线分析与可视化。
- 提供开关和频率控制，支持汇总（全 batch 统计）与按样本少量记录两种模式，尽量零侵入、低开销。

## 集成点（与现有代码对齐）
- 入口：`BimanGrasp-Optimization/main.py` 的 `GraspExperiment.run_optimization()`。
  - 该函数每步已获取 `energy_terms`（含 `total/fc/dis/pen/spen/joints/vew`），也有 MH 接受掩码 `accept` 与当前 `step`。
  - 现有 `Logger` 负责写 TensorBoard 的均值；本方案新增磁盘落盘（CSV/JSONL）。

## 新增配置（utils/config.py）
新增 `MetricsConfig` 并挂到 `ExperimentConfig`：
- `enabled: bool = False`：是否启用指标记录
- `stride: int = 50`：每隔多少步记录一次
- `format: str = 'csv'`：`csv` 或 `jsonl`
- `summary: bool = True`：是否记录 batch 汇总（均值/中位数/分位数）
- `per_sample_num: int = 0`：是否额外记录少量样本的逐步指标，0 为不记录
- `sample_object_index: int = 0`、`sample_local_start: int = 0`：与可视化一致的样本定位方式；记录 `per_sample_num` 个样本（索引递增）
- `fields: List[str]`：要记录的字段（默认：`["total","fc","dis","pen","spen","joints","vew","accept_rate","temperature","step_size"]`）
- `out_dirname: str = 'metrics'`：结果目录名
- `file_basename: str = 'metrics'`：文件前缀（汇总 `metrics_summary.csv`，样本 `metrics_sample_k.csv`）

CLI 映射（`main.py`）：
- `--metrics`（bool）
- `--metrics_stride`、`--metrics_format`、`--metrics_summary`、`--metrics_per_sample_num`
- `--metrics_obj`、`--metrics_local_start`
- `--metrics_fields`（可选，逗号分隔）

## 记录内容（字段定义）
- 基本：`step`
- 能量：`total, fc, dis, pen, spen, joints, vew`
- MALA：`accept_rate`（本步平均接受率）、`temperature`（与可视化中相同计算）、`step_size`（当前步步长）
- 可选：`success_fc/dis/pen`（与 Logger 一致阈值的成功率）

## 新增模块（utils/metrics_recorder.py）
`MetricsRecorder`：管理目录、文件句柄与写入。
- 构造：传入 `results_path`、`config.metrics`、batch 相关信息。
- 方法：
  - `log_summary(step, energy_terms, accept_mask, temperature, step_size)`：落盘均值/中位数/分位数（默认均值即可，后续可扩展）。
  - `log_samples(step, sample_indices, energy_terms, accept_mask, temperature, step_size)`：对选定样本写一行（每样本一个文件）。
  - 支持 CSV（首行写表头）与 JSONL（逐行 JSON）。
- 目录结构：`results/metrics/metrics_summary.csv`、`results/metrics/metrics_sample_0.csv` 等。

## 实现步骤
1) `utils/config.py`
   - 新增 `@dataclass MetricsConfig`，并在 `ExperimentConfig` 增加 `metrics: MetricsConfig` 字段。
   - `update_from_args` 中解析 `--metrics*` 映射，`--metrics_fields` 若存在则 split 到 `List[str]`。

2) 新增 `utils/metrics_recorder.py`
   - 提供 CSV/JSONL 两种实现，自动建目录、写表头。
   - 统一字段顺序（按 `fields`），缺失字段跳过或置空。

3) 修改 `main.py`（`run_optimization`）
   - `setup_logging()` 后，若 `metrics.enabled`，在 `results_path` 下创建 `metrics/`。
   - 进入迭代：`logger.log(...)` 之后、或紧随其后，若 `step % stride == 0` 则：
     - 计算 `accept_rate = accept.float().mean()`；
     - 计算 `temperature` 与 `step_size`（复用与优化器一致的公式或在优化器中提供 getter）。
     - 调用 `metrics_recorder.log_summary(...)`，写 `energy_terms` 的均值；
     - 如果 `per_sample_num > 0`：按 `sample_object_index * batch_size_each + sample_local_start + k` 选样本，写 `metrics_sample_k.*`。
   - 结束时，关闭文件句柄（若采用持久句柄）。

4) 依赖与兼容
   - 仅用 Python 标准库（`csv`/`json`/`os`），不新增依赖。
   - 与现有 TensorBoard `Logger` 并存，不冲突。

## 伪代码示例
```python
# in run_optimization(), after logger.log(...)
if metrics.enabled and (step % metrics.stride == 0):
    accept_rate = accept.float().mean().item()
    # recompute or cache temperature / step_size similarly to optimizer
    temperature = optimizer.initial_temperature * optimizer.cooling_schedule ** (optimizer.step // optimizer.annealing_period)
    step_size   = optimizer.step_size * optimizer.cooling_schedule ** (optimizer.step // optimizer.step_size_period)

    # summary: mean across batch
    summary = {
        'step': step,
        'total': energy_terms.total.mean().item(),
        'fc': energy_terms.force_closure.mean().item(),
        'dis': energy_terms.distance.mean().item(),
        'pen': energy_terms.penetration.mean().item(),
        'spen': energy_terms.self_penetration.mean().item(),
        'joints': energy_terms.joint_limits.mean().item(),
        'vew': energy_terms.wrench_volume.mean().item(),
        'accept_rate': accept_rate,
        'temperature': float(temperature),
        'step_size': float(step_size),
    }
    metrics_recorder.log_summary(summary)

    # per-sample (optional)
    for k in range(metrics.per_sample_num):
        gidx = metrics.sample_object_index * object_model.batch_size_each + metrics.sample_local_start + k
        sample_row = {
            'step': step,
            'total': energy_terms.total[gidx].item(),
            'fc': energy_terms.force_closure[gidx].item(),
            'dis': energy_terms.distance[gidx].item(),
            'pen': energy_terms.penetration[gidx].item(),
            'spen': energy_terms.self_penetration[gidx].item(),
            'joints': energy_terms.joint_limits[gidx].item(),
            'vew': energy_terms.wrench_volume[gidx].item(),
            'accept': bool(accept[gidx].item()),
            'temperature': float(temperature),
            'step_size': float(step_size),
        }
        metrics_recorder.log_sample(k, sample_row)
```

## 使用示例（CLI）
```bash
python BimanGrasp-Optimization/main.py \
  --name metrics_demo \
  --object_code Curver_Storage_Bin_Black_Small \
  --batch_size 32 --num_iterations 200 \
  --metrics --metrics_stride 10 --metrics_format csv --metrics_summary \
  --metrics_per_sample_num 2 --metrics_obj 0 --metrics_local_start 0
```
- 输出：`data/experiments/metrics_demo/results/metrics/metrics_summary.csv` 与 `metrics_sample_0.csv`、`metrics_sample_1.csv`。

## 测试清单
- 开关：`--metrics` 关闭/开启下不产生/产生文件。
- 频率：`--metrics_stride=1/10` 观察行数是否符合预期。
- 维度：`per_sample_num` 不越界（≤ `batch_size_each - sample_local_start`）。
- 兼容：与可视化并行开启，二者互不影响（IO 与性能允许）。
- 完整性：CSV 首行表头正确；JSONL 每行为合法 JSON；数值无 `nan/inf`。

## 后续可选增强
- 记录分位数（p10/p50/p90）与标准差；
- 支持 Parquet（需依赖 pyarrow）便于大规模分析；
- 在 `visualization.py` 中加载 `metrics_sample_k.csv`，叠加曲线与视频帧对齐查看；
- 增加 `accept_streak`、`energy_delta` 等更细粒度诊断指标。
