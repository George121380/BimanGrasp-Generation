## 可规模化数据生成与多GPU并行方案（基于现有 `BimanGrasp-Optimization/main.py`）

### 目标
- 在不修改核心优化逻辑的前提下，实现“按物体配额的多轮生成+合并”的可规模化流程。
- 在多 GPU 服务器上实现并行调度，提升吞吐量与生成速度。
- 保持结果可复现（不同轮次种子不同、可恢复中断）、产物稳定（结构化产出与合并）、临时产物可清理。

---

## 阶段 A：单机单 GPU 的“按配额多轮生成 + 合并”

### 总体思路
在现有 `main.py` 不变的前提下，通过“外部调度器（wrapper）”循环调用 `main.py` 多次，每次设置：
- 唯一的 `--name`（用于区分轮次）
- 唯一的 `--seed`（确保跨轮不同）
- 指定 `--object_code` 与 `--batch_size`、`--num_iterations` 等

每次运行完成后，从该轮结果目录读取 `.npy/.npz` 等产物，进行增量合并；当某个 `object_code` 的累计样本数达到目标配额时，停止对该物体的后续轮次。

### 接口与配置（建议新增给“外部调度器”，先不改 `main.py`）
- 必需：
  - `--object_code_list`：物体列表（可与 `main.py` 一致，或使用外部 JSON/文本）。
  - `--target_count_per_object`：每个物体需要的总样本数（整数）。
  - `--round_batch_size`：每轮 `main.py` 使用的 `--batch_size`（默认可复用现有 `--batch_size` 的值）。
  - `--num_iterations`：传递给 `main.py` 的迭代数。
  - `--seed_base`：种子基数，用于构造各轮/各物体唯一种子。
  - `--exp_name`：基础实验名（外部调度器会派生出每轮的 `--name`）。
- 可选：
  - `--max_rounds`：最多轮数的保险限制（避免意外死循环）。
  - `--merge_stride`：合并频率（例如每完成 N 轮再做一次全量合并，降低 I/O 压力）。
  - `--resume`：启用断点恢复（读取清单继续未完成项）。
  - `--keep_intermediate`：是否保留中间轮次产物（默认只在达标后清理）。

### 目录与命名（兼容当前结果结构）
现有 `main.py` 会将结果写入 `data/experiments/{name}/results/`。为避免修改 `main.py`，采用“分轮命名”的方式：
- 每轮外部调用使用唯一 `--name`: `{exp_name}__{object_code}__r{round_id}`
- 该轮结果目录：`data/experiments/{exp_name}__{object_code}__r{round_id}/results/`
- 合并输出（聚合后的最终结果）建议统一到：
  - `data/experiments/{exp_name}/final/{object_code}/` 下的结构化产物（例如 `final.npz`）
  - 同时在 `data/experiments/{exp_name}/manifest.json` 中维护全局清单（计数、已完成轮次、种子等）

### 种子策略（保证跨轮与跨物体不冲突）
- 令 `object_index` 为物体在列表中的索引，`round_id` 从 0 递增。
- 计算规则：`seed = seed_base + object_index * 100000 + round_id`
  - 避免不同物体/轮次种子碰撞；易追踪与复现。

### 调度伪代码（单 GPU）
```python
for object_index, object_code in enumerate(object_code_list):
    collected = count_already_in_final(exp_name, object_code)  # 读取已合并数量
    round_id = 0
    while collected < target_count_per_object and round_id < max_rounds:
        name = f"{exp_name}__{object_code}__r{round_id}"
        seed = seed_base + object_index * 100000 + round_id
        run_main(name=name, object_code=object_code, batch_size=round_batch_size,
                 num_iterations=num_iterations, seed=seed, gpu="0")

        produced = read_round_results(name)  # 从该轮 results 目录读取样本
        collected = merge_into_final(exp_name, object_code, produced,
                                     target_count_per_object)

        if not keep_intermediate and collected >= target_count_per_object:
            cleanup_round_results(name)

        round_id += 1
    mark_object_done_in_manifest(exp_name, object_code, collected)
```

### 合并策略与产物格式
- 建议最终以 `.npz` 打包，键名覆盖主要实体：
  - `samples`（或 `poses`）：手/物体位姿或抓取参数，按样本维拼接。
  - `energies`：能量项（`total`, `distance`, `penetration`, `self_penetration`, `joint_limits`, `wrench_volume`）。
  - `meta`：包含 `seed`、`round_id`、`name`、`timestamp` 等源信息（可另存 `manifest.json`）。
- 合并采用 `np.concatenate`；当累计超过 `target_count_per_object` 时，进行截断（只保留前 `target_count_per_object` 个）。
- 质量过滤（可选）：若某轮保存了能量/阈值，可在合并时丢弃明显不合格样本（如 `penetration` 超阈值）。
- 形状校验：不同轮次产物必须字段一致、shape 兼容；若缺字段则记录并跳过该轮对应字段。

### 断点恢复（`--resume`）
- `manifest.json` 记录：
  - 每个 `object_code` 的累计计数、完成轮次、使用的种子集合
  - 各轮 `name` 与对应结果路径、合并状态
- 断点恢复时先读取 `manifest.json`，跳过已完成任务，仅继续未达标物体。

### 临时文件清理
- 当某物体达标后：可清理该物体对应轮次的 `frames/`、`optimization_*.mp4`、中间 `.npy`，只保留 `final/` 与 `manifest.json`。
- 若需保留可追溯性，可只清理渲染帧与可重构文件，保留每轮的 `config.txt` 与关键 `.npz`。

---

## 阶段 B：单机多 GPU 并行调度

### 并行模型
- 将“任务”抽象为：某 `object_code` 的一轮运行（或多个小轮）。
- 将“工作者”抽象为：一个 GPU 进程。每个工作者独占一个 GPU。
- 外部调度器维护任务队列，空闲 GPU 拉取下一任务并运行 `main.py`。

### GPU 绑定策略
- 通过子进程环境变量隔离：
  - `CUDA_VISIBLE_DEVICES={gpu_id}`（使得子进程内只见到一块 GPU）
  - 同时传入 `--gpu 0` 给 `main.py`（与现有设备选择逻辑兼容）
- 控制 CPU 线程占用以减少抖动：
  - 设置 `OMP_NUM_THREADS`, `MKL_NUM_THREADS` 为合理值（如 1-4）。

### 并发调度伪代码
```python
gpu_ids = parse_gpus("0,1,2,3")
task_queue = build_tasks(object_code_list, target_count_per_object, round_batch_size)

with ProcessPool(max_workers=len(gpu_ids)) as pool:
    futures = []
    for task in task_queue:
        gpu_id = next_free_gpu(gpu_ids)  # 简单轮转或基于可用性
        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        futures.append(pool.submit(run_one_round, task, env))

    for f in as_completed(futures):
        produced = f.result()
        merge_into_final(...)
        update_manifest(...)
```

### 种子与唯一性
- 在阶段 A 的规则上附加 GPU 维度不改变种子（即保持与单 GPU 情况一致），确保同一任务无论分配到哪个 GPU，都能复现相同结果。

### 失败重试与容错
- 子进程异常或返回码非 0：记录到 `logs/gpu_{id}.log` 并重试 N 次（如 2 次）。
- 若轮次失败次数超过阈值：跳过该轮，继续其他任务并在 `manifest.json` 标注。

### 合并与清理
- 多 GPU 并行期间，合并过程需要加锁（文件级或进程级）：
  - 简单实现：合并只在主进程执行；工作进程仅负责产出轮次结果。
- 达标后触发清理策略（同阶段 A）。

---

## 阶段 C：后续可选的代码内建改造（迭代方向）
> 本阶段为未来迭代方向，当前仅供参考，暂不修改代码。

1) 在 `main.py` 内部支持“轮次循环 + 合并 + 达标即停”的原生模式：
- 新参数：`--target_count_per_object`、`--resume`、`--seed_base` 等
- 内部构造轮次 `name`、自动保存 `manifest.json`、并原地合并

2) 原生多 GPU：
- 使用 `torch.multiprocessing` 或 `torch.distributed` 启动多个进程，每个进程绑定一张 GPU，内部共享任务队列。

3) 结果格式统一：
- 统一使用 `.npz` 保存，并定义固定键集合与 dtype，降低后处理复杂度。

---

## 命令与操作示例（在无代码改动前）

### 单 GPU 外部调度（示意）
```bash
python tools/scale_runner.py \
  --exp_name grasp_bigrun \
  --object_code_list Curver_Storage_Bin_Black_Small Hasbro_Monopoly_Hotels_Game \
  --target_count_per_object 10000 \
  --round_batch_size 512 \
  --num_iterations 10000 \
  --seed_base 12345 \
  --resume
```

### 多 GPU 外部调度（示意）
```bash
python tools/scale_runner.py \
  --exp_name grasp_bigrun \
  --gpus 0,1,2,3 \
  --object_code_list Curver_Storage_Bin_Black_Small Hasbro_Monopoly_Hotels_Game \
  --target_count_per_object 10000 \
  --round_batch_size 512 \
  --num_iterations 10000 \
  --seed_base 12345 \
  --resume
```

> 说明：`tools/scale_runner.py` 为建议新增的外部调度脚本（后续实现）。当前方案文档仅描述行为，不修改现有代码。

---

## 验收标准
- 每个 `object_code` 的最终样本数精确达到（或截断至）`target_count_per_object`。
- 各轮使用不同种子，可复现且清晰记录在 `manifest.json`。
- 失败轮次被记录并可重试；中断后可 `--resume` 继续。
- 多 GPU 并发下无数据竞态（合并在主进程完成或有锁保护）。
- 达标后中间产物按策略清理，磁盘空间可控。

---

## 风险与注意事项
- 不同轮次可能存在重复或近似样本：必要时在合并阶段引入“位姿/能量阈值去重”。
- I/O 压力：大量帧/视频的写入影响吞吐，可在规模化生成时关闭可视化或降低 `--vis_frame_stride`。
- 显存/内存占用：多 GPU 并发时注意 `batch_size` 上限与 `num_workers` 配置，避免 OOM。
- 日志与监控：建议每轮写入 `config.txt` 与简要 `metrics`，便于异常排查与复现。

---

## 实施里程碑
- 里程碑 1：外部调度器（单 GPU）+ 合并器 + 清单与恢复（0.5–1 天）
- 里程碑 2：多 GPU 调度与隔离、失败重试、主进程合并（0.5–1 天）
- 里程碑 3（可选）：内建到 `main.py` 的原生多轮/多 GPU 支持（1–2 天）


