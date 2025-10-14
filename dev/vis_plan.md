# 优化过程视频记录方案（BimanGrasp-Generation）

## 目标
- 在优化迭代过程中，周期性捕获双手与物体的三维画面，并导出为视频（mp4）。
- 非数据曲线，而是“可看到两只手逐渐靠近并抓住物体”的可视化动画。

## 约束与考虑
- 批量优化：同一次运行包含多个对象与样本（`total_batch_size = len(object_code_list) * model.batch_size`）。需要指定“拍摄的样本索引”。
- 计算/存储成本：默认迭代 10k 次，不宜每步都渲染，建议步进采样（如每 50～100 步截帧）。
- 渲染后端：复用现有 `plotly` 可视化（`HandModel.get_plotly_data`、`ObjectModel.get_plotly_data`），用 `kaleido` 导出静态图片；再用 `imageio` 或 `ffmpeg` 合成视频。
- 视角一致：固定相机与场景参数，保证视频稳定不抖动。

## 集成点（现有代码）
- `BimanGrasp-Optimization/main.py`：
  - `GraspExperiment.run_optimization()` 为迭代主循环，适合插入“按步保存帧”。
  - `setup_logging()` 已创建实验目录，可在其中创建 `frames/` 子目录。
- `utils/hand_model.py`：
  - `HandModel.get_plotly_data(i, ...)` 可输出单手的 Plotly Mesh3d 与接触点点云。
- `utils/object_model.py`：
  - `ObjectModel.get_plotly_data(i, ...)` 可输出物体网格。

## 新增组件与配置
1) 新增可视化配置 `VisConfig`（添加到 `utils/config.py` 并纳入 `ExperimentConfig`）：
- `enabled: bool = False`
- `sample_object_index: int = 0`（对象索引）
- `sample_local_index: int = 0`（该对象内的样本索引）
- `frame_stride: int = 50`（每多少步保存一帧）
- `fps: int = 30`
- `width: int = 900`, `height: int = 900`
- `out_dirname: str = 'frames'`（帧目录名）
- `video_filename: str = 'optimization.mp4'`
- `show_contacts: bool = False`（是否显示接触点）
- `bg_color: str = '#E2F0D9'`（与 `visualization.py` 保持一致）

并在 `ExperimentConfig.update_from_args` 支持 CLI 参数：
- `--vis`, `--vis_frame_stride`, `--vis_obj`, `--vis_local`, `--vis_fps`, `--vis_width`, `--vis_height`, `--vis_contacts`。

2) 新增 `utils/visualization_recorder.py`（建议文件名）：
- `FrameRecorder` 类职责：
  - 初始化时绑定：`object_model`、`bimanual_pair`、输出目录、图像尺寸、相机/背景配置。
  - 计算“全局样本索引”：`global_idx = sample_object_index * batch_size_each + sample_local_index`。
  - `capture(step: int)`: 组装 `left_hand.get_plotly_data(global_idx)`、`object.get_plotly_data(global_idx)`、`right_hand.get_plotly_data(global_idx)`，构建单一 `go.Figure`；设置背景、隐藏坐标轴、固定 `scene_aspectmode='data'` 与相机；调用 `fig.write_image(path, width, height)`（依赖 `kaleido`）。
  - `finalize(video_path: str, fps: int)`: 使用 `imageio` 读取按步保存的 PNG 序列，合成为 mp4（或调用系统 `ffmpeg`）。

## 代码改动点（高层步骤）
1) `utils/config.py`
- 新增 `@dataclass class VisConfig`。
- 在 `ExperimentConfig` 中新增字段 `vis: VisConfig = field(default_factory=VisConfig)`。
- 在 `update_from_args` 中解析 `--vis*` 相关参数。

2) 新增 `utils/visualization_recorder.py`
- 实现 `FrameRecorder`（见下方伪代码）。

3) 修改 `BimanGrasp-Optimization/main.py`
- `GraspExperiment.setup_logging()`：
  - 若 `config.vis.enabled`，创建 `frames_dir = os.path.join(results_path, config.vis.out_dirname)`。
- `GraspExperiment.run_optimization()`：
  - 在初始能量计算后，若开启可视化，先 `recorder.capture(step=0)`。
  - 每步结束（完成 MH 决策、记录日志后），若 `step % frame_stride == 0`，调用 `recorder.capture(step)`。
- `GraspExperiment.save_final_results()` 之后：
  - 若开启可视化，`recorder.finalize(video_output_path, fps)`。

4) 依赖与安装
- 增加：`pip install -U kaleido imageio[ffmpeg]`
- 或在系统安装 `ffmpeg` 后仅使用 `imageio`：`imageio.get_writer('...mp4', fps=fps)`。
- 可在 `install.sh` 末尾追加可选安装提示。

## 伪代码示例
- 计算全局索引：
```python
# global_idx 计算（在 recorder 内部或 main 中）
# 注意：这里的 batch_size_each 来源于 object_model.config 或 config.model.batch_size_each
# 若 main 使用 config.model.batch_size（每对象抓取样本数），与 ObjectModel(batch_size_each) 保持一致

global_idx = vis.sample_object_index * config.model.batch_size_each + vis.sample_local_index
```

- `FrameRecorder.capture`（简化示意）：
```python
import plotly.graph_objects as go

class FrameRecorder:
    def __init__(self, object_model, bimanual_pair, global_idx, frames_dir,
                 width=900, height=900, show_contacts=False, bg_color="#E2F0D9"):
        self.object_model = object_model
        self.bimanual_pair = bimanual_pair
        self.global_idx = global_idx
        self.frames_dir = frames_dir
        self.width = width
        self.height = height
        self.show_contacts = show_contacts
        self.bg_color = bg_color
        os.makedirs(self.frames_dir, exist_ok=True)

    def _build_figure(self):
        left_traces = self.bimanual_pair.left.get_plotly_data(
            i=self.global_idx, opacity=1.0, color='lightslategray',
            with_contact_points=self.show_contacts
        )
        right_traces = self.bimanual_pair.right.get_plotly_data(
            i=self.global_idx, opacity=1.0, color='lightslategray',
            with_contact_points=self.show_contacts
        )
        obj_traces = self.object_model.get_plotly_data(
            i=self.global_idx, color='seashell', opacity=1.0
        )
        fig = go.Figure(right_traces + obj_traces + left_traces)
        fig.update_layout(paper_bgcolor=self.bg_color, plot_bgcolor=self.bg_color)
        fig.update_layout(scene_aspectmode='data')
        fig.update_layout(scene=dict(
            xaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            yaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False),
            zaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showticklabels=False)
        ))
        return fig

    def capture(self, step: int):
        fig = self._build_figure()
        out_path = os.path.join(self.frames_dir, f"frame_{step:06d}.png")
        fig.write_image(out_path, width=self.width, height=self.height)

    def finalize(self, video_path: str, fps: int = 30):
        import imageio
        frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        with imageio.get_writer(video_path, fps=fps) as writer:
            for fname in frames:
                writer.append_data(imageio.v2.imread(os.path.join(self.frames_dir, fname)))
```

- 在 `run_optimization()` 中调用（概念示例）：
```python
# 初始化后
if config.vis.enabled:
    global_idx = config.vis.sample_object_index * config.model.batch_size_each + config.vis.sample_local_index
    frames_dir = os.path.join(results_path, config.vis.out_dirname)
    recorder = FrameRecorder(object_model, bimanual_pair, global_idx, frames_dir,
                             width=config.vis.width, height=config.vis.height,
                             show_contacts=config.vis.show_contacts)
    recorder.capture(step=0)

for step in range(1, config.optimizer.num_iterations + 1):
    # Langevin 提案 + 计算能量 + MH 接受 ...
    if config.vis.enabled and (step % config.vis.frame_stride == 0):
        recorder.capture(step)

# 结束优化
if config.vis.enabled:
    video_path = os.path.join(results_path, config.vis.video_filename)
    recorder.finalize(video_path, fps=config.vis.fps)
```

## 输出与目录结构
- 帧图像：`data/experiments/<name>/results/frames/frame_000000.png, frame_000050.png, ...`
- 视频：`data/experiments/<name>/results/optimization.mp4`

## 相机与美观建议
- 维持与 `visualization.py` 一致的背景与隐藏坐标轴设置，确保主体聚焦。
- 可选：设置固定 `scene_camera`（距离/方位）以获得稳定视角；如需围绕物体旋转相机，可在每帧施加小幅度轨迹，但默认保持固定。

## 性能与稳定性
- 建议 `frame_stride >= 20`，`width/height <= 1024` 以控制开销。
- 若 `kaleido` 导出失败，检查环境：`pip install -U kaleido`；Linux 服务器通常无需额外依赖。
- Windows 下若使用 `ffmpeg`，确保可执行在 `PATH` 中；否则使用纯 `imageio` 路径。

## 使用示例（CLI）
```bash
# 仅示例；最终以参数对接为准
python BimanGrasp-Optimization/main.py \
  --name vis_demo \
  --vis True \
  --vis_frame_stride 50 \
  --vis_obj 0 \
  --vis_local 0 \
  --vis_fps 30
```

## 测试清单
- 小迭代数（如 200 次）快速回归：确认能生成若干帧与 mp4 文件。
- 多对象/多样本：验证 `global_idx` 计算正确（画面对象、尺度与期望一致）。
- 边界：`sample_local_index` 超界应报错或回退；`kaleido` 缺失应给出清晰提示。
- 性能：在 A40 上以默认 stride 渲染 10k 步，估计总耗时与磁盘占用。
