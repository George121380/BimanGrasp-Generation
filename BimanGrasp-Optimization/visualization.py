import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
import plotly.io as pio
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.config import HandSpec

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='Curver_Storage_Bin_Black_Small')
    parser.add_argument('--num', type=int, default=0)
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
    # load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    
    right_qpos = data_dict['qpos_right']
    right_rot = np.array(transforms3d.euler.euler2mat(*[right_qpos[name] for name in rot_names]))
    right_rot = right_rot[:, :2].T.ravel().tolist()
    # Order joint angles using the model's joint names (dynamic DOF)
    right_joint_names = getattr(right_hand_model, 'joints_names', [])
    right_joint_values = [right_qpos.get(name, 0.0) for name in right_joint_names]
    right_hand_pose = torch.tensor([right_qpos[name] for name in translation_names] + right_rot + right_joint_values, dtype=torch.float, device=device)
    if 'qpos_right_st' in data_dict:
        right_qpos_st = data_dict['qpos_right_st']
        right_rot = np.array(transforms3d.euler.euler2mat(*[right_qpos_st[name] for name in rot_names]))
        right_rot = right_rot[:, :2].T.ravel().tolist()
        right_joint_values_st = [right_qpos_st.get(name, 0.0) for name in right_joint_names]
        right_hand_pose_st = torch.tensor([right_qpos_st[name] for name in translation_names] + right_rot + right_joint_values_st, dtype=torch.float, device=device)

    # load left results
    left_qpos = data_dict['qpos_left']
    left_rot = np.array(transforms3d.euler.euler2mat(*[left_qpos[name] for name in rot_names]))
    left_rot = left_rot[:, :2].T.ravel().tolist()
    left_joint_names = getattr(left_hand_model, 'joints_names', [])
    left_joint_values = [left_qpos.get(name, 0.0) for name in left_joint_names]
    left_hand_pose = torch.tensor([left_qpos[name] for name in translation_names] + left_rot + left_joint_values, dtype=torch.float, device=device)
    if 'qpos_left_st' in data_dict:
        left_qpos_st = data_dict['qpos_left_st']
        left_rot = np.array(transforms3d.euler.euler2mat(*[left_qpos_st[name] for name in rot_names]))
        left_rot = left_rot[:, :2].T.ravel().tolist()
        left_joint_values_st = [left_qpos_st.get(name, 0.0) for name in left_joint_names]
        left_hand_pose_st = torch.tensor([left_qpos_st[name] for name in translation_names] + left_rot + left_joint_values_st, dtype=torch.float, device=device)


    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    right_hand_model.set_parameters(right_hand_pose.unsqueeze(0))
    right_hand_en_plotly = right_hand_model.get_plotly_data(i=0, opacity=1, color='lightslategray', with_contact_points=False)
    
    left_hand_model.set_parameters(left_hand_pose.unsqueeze(0))
    left_hand_en_plotly = left_hand_model.get_plotly_data(i=0, opacity=1, color='lightslategray', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='seashell', opacity=1)    
       
    
    fig = go.Figure(right_hand_en_plotly + object_plotly + left_hand_en_plotly)

    fig.update_layout(
        paper_bgcolor='#E2F0D9',  
        plot_bgcolor='#E2F0D9'    
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
