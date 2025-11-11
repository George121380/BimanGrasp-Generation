import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
import plotly.io as pio
from itertools import cycle
from plotly.colors import qualitative as plotly_qualitative
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.config import HandSpec

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']


def _parse_color_string(color_str):
    """Convert common color string formats to RGB tuple."""
    if not isinstance(color_str, str):
        raise ValueError("Color value must be a string.")
    s = color_str.strip()
    if s.startswith('#'):
        hex_value = s[1:]
        if len(hex_value) == 3:
            hex_value = ''.join([ch * 2 for ch in hex_value])
        if len(hex_value) != 6:
            raise ValueError(f"Unsupported hex color format: {color_str}")
        return tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4))
    if s.startswith('rgb'):
        values = s[s.find('(') + 1:s.find(')')].split(',')
        rgb = tuple(int(float(v.strip())) for v in values[:3])
        if len(rgb) != 3:
            raise ValueError(f"Unsupported rgb color format: {color_str}")
        return rgb
    raise ValueError(f"Unsupported color format: {color_str}")


def _rgb_to_hex(rgb):
    """Convert RGB tuple (0-255) back to hex string."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def _lighten_color(color_str, amount=0.6):
    """Lighten a color by blending it towards white."""
    try:
        r, g, b = _parse_color_string(color_str)
    except ValueError:
        return color_str
    amount = float(np.clip(amount, 0.0, 1.0))
    lightened = (
        int(r + (255 - r) * amount),
        int(g + (255 - g) * amount),
        int(b + (255 - b) * amount)
    )
    return _rgb_to_hex(lightened)


def _collect_frame_files(result_path, object_code):
    """Collect result files across experiment directories for timeline visualization."""
    import re

    result_dir = os.path.abspath(result_path)
    base_file = os.path.join(result_dir, f"{object_code}.npy")

    def sort_key(name):
        match = re.match(r'^-?\d+$', name)
        if match:
            return (0, int(name))
        return (1, name)

    frame_candidates = {}

    # Traverse sibling experiment directories (e.g., ../0/results, ../1/results, ...)
    exp_dir = os.path.dirname(result_dir)
    root_dir = os.path.dirname(exp_dir)
    if os.path.isdir(root_dir):
        try:
            entries = os.listdir(root_dir)
        except OSError:
            entries = []
        for entry in entries:
            candidate_results = os.path.join(root_dir, entry, 'results')
            candidate_file = os.path.join(candidate_results, f"{object_code}.npy")
            if os.path.isfile(candidate_file):
                frame_candidates[candidate_file] = (sort_key(entry), candidate_results)

    # Ensure the explicitly provided result directory is included
    exp_entry = os.path.basename(os.path.dirname(result_dir))
    if os.path.isfile(base_file):
        frame_candidates.setdefault(base_file, (sort_key(exp_entry), result_dir))

    if not frame_candidates:
        return [(result_dir, base_file)]

    sorted_frames = sorted(frame_candidates.items(), key=lambda item: item[1][0])
    return [(directory, file_path) for file_path, (_, directory) in sorted_frames]


def _qpos_to_components(qpos_dict, joint_names):
    """Extract translation, rotation (flattened matrix cols) and joint values."""
    rot_matrix = transforms3d.euler.euler2mat(*[qpos_dict[name] for name in rot_names])
    rot_cols = rot_matrix[:, :2].T.ravel().tolist()
    joint_values = [qpos_dict.get(name, 0.0) for name in joint_names]
    translation = [qpos_dict[name] for name in translation_names]
    return translation, rot_cols, joint_values


def _hand_pose_tensor(qpos_dict, joint_names, device):
    translation, rot_cols, joint_values = _qpos_to_components(qpos_dict, joint_names)
    return torch.tensor(translation + rot_cols + joint_values, dtype=torch.float, device=device)


def _pose_feature_vector(data_dict, right_joint_names, left_joint_names):
    """Generate a numpy feature vector describing both hands for distance computation."""
    r_translation, r_rot_cols, r_joint_values = _qpos_to_components(data_dict['qpos_right'], right_joint_names)
    l_translation, l_rot_cols, l_joint_values = _qpos_to_components(data_dict['qpos_left'], left_joint_names)
    return np.array(r_translation + r_rot_cols + r_joint_values + l_translation + l_rot_cols + l_joint_values, dtype=np.float32)


def _farthest_point_sampling(features, target_count):
    """Greedy farthest point sampling on feature vectors."""
    if target_count >= len(features):
        return list(range(len(features)))
    norms = np.linalg.norm(features, axis=1)
    first_index = int(np.argmax(norms))
    selected = [first_index]
    distances = np.linalg.norm(features - features[first_index], axis=1)
    for _ in range(1, target_count):
        next_index = int(np.argmax(distances))
        selected.append(next_index)
        new_distances = np.linalg.norm(features - features[next_index], axis=1)
        distances = np.minimum(distances, new_distances)
    return selected

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='Curver_Storage_Bin_Black_Small')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--nums', type=int, nargs='+', default=None)
    parser.add_argument('--count', type=int, default=None)
    parser.add_argument('--no_st', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--result_path', type=str, default='../data/experiments/test/results')
    parser.add_argument('--renderer', type=str, default='browser')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_path', type=str, default=None)
    # Hand presets: shadow | psi_oy (can override per side)
    parser.add_argument('--hand', type=str, default=None)
    parser.add_argument('--left_hand', type=str, default=None)
    parser.add_argument('--right_hand', type=str, default=None)

    args = parser.parse_args()

    device = 'cpu'
    # Ensure an interactive renderer (e.g., browser) is used
    try:
        pio.renderers.default = args.renderer
    except Exception:
        pass

    # Resolve hand presets
    def choose_spec(choice, side):
        if choice is None:
            return HandSpec.preset_shadow('left' if side == 'left' else 'right')
        c = str(choice).lower()
        if c in ['shadow', 'shadowhand', 'mjcf']:
            return HandSpec.preset_shadow('left' if side == 'left' else 'right')
        if c in ['psi_oy', 'psi-oy', 'psioy', 'urdf']:
            return HandSpec.preset_psi_oy('left' if side == 'left' else 'right')
        return HandSpec.preset_shadow('left' if side == 'left' else 'right')

    left_choice = args.left_hand if args.left_hand is not None else args.hand
    right_choice = args.right_hand if args.right_hand is not None else args.hand
    lh_spec = choose_spec(left_choice, 'left')
    rh_spec = choose_spec(right_choice, 'right')

    # Build models from specs
    left_hand_model = HandModel(
        mjcf_path=lh_spec.model_path if lh_spec.model_format == 'mjcf' else None,
        urdf_path=lh_spec.model_path if lh_spec.model_format == 'urdf' else None,
        mesh_path=lh_spec.mesh_path,
        contact_points_path=lh_spec.contact_points_path,
        penetration_points_path=lh_spec.penetration_points_path,
        device=device,
        handedness='left_hand',
        exclude_links_for_sdf=lh_spec.exclude_links_for_sdf if hasattr(lh_spec, 'exclude_links_for_sdf') else None
    )

    right_hand_model = HandModel(
        mjcf_path=rh_spec.model_path if rh_spec.model_format == 'mjcf' else None,
        urdf_path=rh_spec.model_path if rh_spec.model_format == 'urdf' else None,
        mesh_path=rh_spec.mesh_path,
        contact_points_path=rh_spec.contact_points_path,
        penetration_points_path=rh_spec.penetration_points_path,
        device=device,
        handedness='right_hand',
        exclude_links_for_sdf=rh_spec.exclude_links_for_sdf if hasattr(rh_spec, 'exclude_links_for_sdf') else None
    )
    
    object_model = ObjectModel(
        data_root_path='../data/meshdata',
        batch_size_each=1,
        num_samples=2000, 
        device=device,
        size='large'
    )

    # Collect frame files (for time-series visualization) and load base dataset
    frame_sources = _collect_frame_files(args.result_path, args.object_code)
    dataset_cache = {}
    index_map_cache = {}

    def get_dataset(file_path):
        if file_path not in dataset_cache:
            dataset_cache[file_path] = np.load(file_path, allow_pickle=True)
        return dataset_cache[file_path]
    
    def get_index_map(file_path):
        if file_path in index_map_cache:
            return index_map_cache[file_path]
        ds = get_dataset(file_path)
        item_to_idx = {}
        try:
            n = len(ds)
        except Exception:
            n = 0
        for i in range(n):
            try:
                entry = ds[i].item() if isinstance(ds[i], np.ndarray) else ds[i]
            except Exception:
                entry = ds[i]
            try:
                pose = _to_pose_dict(entry)
            except Exception:
                continue
            meta = pose.get('__meta__', {})
            item_idx = meta.get('item_index', None)
            if item_idx is not None and item_idx not in item_to_idx:
                item_to_idx[item_idx] = i
        index_map_cache[file_path] = item_to_idx
        return item_to_idx

    dataset = get_dataset(frame_sources[0][1])
    total_poses = len(dataset)
    indices = None
    if args.nums is not None:
        indices = args.nums
    elif args.count is not None:
        if args.count <= 0:
            raise ValueError("--count must be a positive integer.")
    else:
        indices = [args.num]

    if indices is not None:
        if len(indices) == 0:
            raise ValueError("No pose indices provided. Use --num or --nums to specify poses.")
        validated_indices = []
        for idx in indices:
            if idx < 0 or idx >= total_poses:
                raise ValueError(f"Pose index {idx} is out of range for dataset of size {total_poses}.")
            validated_indices.append(idx)
    else:
        count = min(args.count, total_poses)
        if count <= 0:
            raise ValueError("Requested pose count is zero after validation.")

    def _to_pose_dict(entry):
        if isinstance(entry, dict):
            return entry
        if isinstance(entry, np.ndarray) and entry.shape == ():
            value = entry.item()
            if isinstance(value, dict):
                return value
        raise TypeError(f"Unsupported pose data type: {type(entry)}. Expect dict-like entries.")

    right_joint_names = getattr(right_hand_model, 'joints_names', [])
    left_joint_names = getattr(left_hand_model, 'joints_names', [])

    if indices is None:
        feature_vectors = []
        for entry in dataset:
            pose_dict = _to_pose_dict(entry)
            feature_vectors.append(_pose_feature_vector(pose_dict, right_joint_names, left_joint_names))
        feature_matrix = np.vstack(feature_vectors)
        validated_indices = _farthest_point_sampling(feature_matrix, count)

    # Initialize object once
    reference_dict = _to_pose_dict(dataset[validated_indices[0]])
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(reference_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    # Prepare figure data
    figure_traces = []
    object_traces = object_model.get_plotly_data(i=0, color='#b0b0b0', opacity=1)
    for trace in object_traces:
        trace.name = 'object'
        trace.showlegend = False
    figure_traces.extend(object_traces)

    color_cycle = cycle(plotly_qualitative.Dark24 if hasattr(plotly_qualitative, 'Dark24') else plotly_qualitative.Plotly)

    primary_pose_indices = set()
    if indices is None and len(validated_indices) >= 2:
        primary_pose_indices = set(validated_indices[:2])

    for pose_idx in validated_indices:
        # Resolve base pose for meta mapping
        try:
            base_pose_dict = _to_pose_dict(dataset[pose_idx])
        except Exception:
            continue
        base_meta = base_pose_dict.get('__meta__', {})
        base_item_index = base_meta.get('item_index', None)
        base_palette_color = next(color_cycle)

        target_frames = frame_sources if pose_idx in primary_pose_indices else frame_sources[:1]
        frames_count = len(target_frames)

        for frame_order, (_, frame_file) in enumerate(target_frames):
            dataset_frame = get_dataset(frame_file)
            # Prefer mapping by item_index across frames; fallback to same index
            frame_idx = None
            if base_item_index is not None:
                frame_map = get_index_map(frame_file)
                frame_idx = frame_map.get(base_item_index, None)
            if frame_idx is None:
                if pose_idx >= len(dataset_frame):
                    continue
                frame_idx = pose_idx
            try:
                frame_data_dict = _to_pose_dict(dataset_frame[frame_idx])
            except Exception:
                continue

            if frames_count > 1:
                lighten_amount = max(0.25, 0.9 - frame_order * 0.18)
            else:
                lighten_amount = 0.9
            pose_color = _lighten_color(base_palette_color, amount=lighten_amount)
            right_hand_pose = _hand_pose_tensor(frame_data_dict['qpos_right'], right_joint_names, device)
            left_hand_pose = _hand_pose_tensor(frame_data_dict['qpos_left'], left_joint_names, device)

            # Adjust opacity: primary pose fades with frame order, others stay consistent
            if pose_idx in primary_pose_indices:
                right_opacity = max(0.2, 0.9 - frame_order * 0.1)
                left_opacity = max(0.15, 0.6 - frame_order * 0.08)
            else:
                right_opacity = 0.9
                left_opacity = 0.6

            right_hand_model.set_parameters(right_hand_pose.unsqueeze(0))
            right_traces = right_hand_model.get_plotly_data(i=0, opacity=right_opacity, color=pose_color, with_contact_points=False)
            for trace in right_traces:
                trace.name = f'right_hand_{pose_idx}' + (f'_f{frame_order}' if frames_count > 1 else '')
                trace.showlegend = frame_order == 0
            figure_traces.extend(right_traces)

            left_hand_model.set_parameters(left_hand_pose.unsqueeze(0))
            left_traces = left_hand_model.get_plotly_data(i=0, opacity=left_opacity, color=pose_color, with_contact_points=False)
            for trace in left_traces:
                trace.name = f'left_hand_{pose_idx}' + (f'_f{frame_order}' if frames_count > 1 else '')
                trace.showlegend = frame_order == 0
            figure_traces.extend(left_traces)

    fig = go.Figure(figure_traces)

    fig.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF'
    )

    fig.update_layout(scene_aspectmode='data')    
    fig.update_layout(
    scene=dict(
        xaxis=dict(
            visible=False,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False
        ),
        zaxis=dict(
            visible=False,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False
        )
    )
    )
    
    # Save to HTML or show in browser
    if bool(getattr(args, 'save', False)):
        # Determine output path
        if args.save_path is not None and len(args.save_path) > 0:
            out_path = args.save_path
        else:
            # default to result_path/object_code_num.html
            base_name = f"{args.object_code}_{args.num}.html"
            out_path = os.path.join(args.result_path, base_name)
        try:
            import plotly.io as pio
            pio.write_html(fig, file=out_path, include_plotlyjs='cdn', full_html=True, auto_open=False)
        except Exception:
            fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)
        print(f"Saved visualization to: {out_path}")
    else:
        fig.show()


# python visualization.py --object_code 100015 --count 6 --save --result_path ../data/experiments/10/results