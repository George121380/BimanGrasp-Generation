"""
Configuration management for bimanual grasp generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
import math
import os


# Global Constants - Joint and Transform Names
JOINT_NAMES = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
TRANSLATION_NAMES = ['WRJTx', 'WRJTy', 'WRJTz']
ROTATION_NAMES = ['WRJRx', 'WRJRy', 'WRJRz']

# Default Joint Angles for Initialization
LEFT_HAND_JOINT_MU = [0.1, 0, -0.6, 0, 0, 0, -0.6, 0, -0.1, 0, -0.6, 0, 0, -0.2, 0, -0.6, 0, 0, -1.2, 0, -0.2, 0]
RIGHT_HAND_JOINT_MU = [0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0]

@dataclass
class PathConfig:
    """File and directory path configuration."""
    
    # MJCF and mesh paths
    right_hand_mjcf: str = 'mjcf/right_shadow_hand.xml'
    left_hand_mjcf: str = 'mjcf/left_shadow_hand.xml'
    mesh_path: str = 'mjcf/meshes'
    right_contact_points: str = 'mjcf/right_hand_contact_points.json'
    left_contact_points: str = 'mjcf/left_hand_contact_points.json'
    penetration_points: str = 'mjcf/penetration_points.json'
    
    # Data paths
    data_root_path: str = '../data/meshdata'
    experiments_base: str = '../data/experiments'
    results_base: str = 'data/graspdata'
    
    @property
    def experiment_path(self) -> str:
        return self.experiments_base
    
    def get_experiment_logs_path(self, exp_name: str) -> str:
        return os.path.join(self.experiments_base, exp_name, 'logs')
    
    def get_experiment_results_path(self, exp_name: str) -> str:
        return os.path.join(self.experiments_base, exp_name, 'results')


@dataclass
class HandSpec:
    """Description of a hand model to load (supports MJCF or URDF)."""
    model_format: Literal['mjcf', 'urdf'] = 'mjcf'
    model_path: str = ''
    mesh_path: str = ''
    contact_points_path: Optional[str] = None
    penetration_points_path: Optional[str] = None
    exclude_links_for_sdf: List[str] = field(default_factory=list)
    joint_mu: Optional[List[float]] = None  # Optional mean joint angles for init
    joint_mu_overrides: Dict[str, float] = field(default_factory=dict)  # regex -> bias in [0,1]

    def resolve_paths(self, base_dir: str = '.') -> 'HandSpec':
        # Optionally resolve relative paths later if needed
        return self

    @staticmethod
    def preset_shadow(side: Literal['left', 'right']) -> 'HandSpec':
        if side == 'left':
            return HandSpec(
                model_format='mjcf',
                model_path='mjcf/left_shadow_hand.xml',
                mesh_path='mjcf/meshes',
                contact_points_path='mjcf/left_hand_contact_points.json',
                penetration_points_path='mjcf/penetration_points.json',
                # Keep legacy exclusions for ShadowHand analytics speed
                exclude_links_for_sdf=[
                    'robot0:forearm', 'robot0:wrist_child', 'robot0:ffknuckle_child',
                    'robot0:mfknuckle_child', 'robot0:rfknuckle_child', 'robot0:lfknuckle_child',
                    'robot0:thbase_child', 'robot0:thhub_child'
                ]
            )
        else:
            return HandSpec(
                model_format='mjcf',
                model_path='mjcf/right_shadow_hand.xml',
                mesh_path='mjcf/meshes',
                contact_points_path='mjcf/right_hand_contact_points.json',
                penetration_points_path='mjcf/penetration_points.json',
                exclude_links_for_sdf=[
                    'robot0:forearm', 'robot0:wrist_child', 'robot0:ffknuckle_child',
                    'robot0:mfknuckle_child', 'robot0:rfknuckle_child', 'robot0:lfknuckle_child',
                    'robot0:thbase_child', 'robot0:thhub_child'
                ]
            )

    @staticmethod
    def preset_psi_oy(side: Literal['left', 'right']) -> 'HandSpec':
        if side == 'left':
            return HandSpec(
                model_format='urdf',
                model_path='psi-oy/InspireHand_OY_Left/InspireHand_OY_Left.urdf',
                mesh_path='psi-oy/InspireHand_OY_Left/meshes',
                contact_points_path=None,
                penetration_points_path=None,
                exclude_links_for_sdf=[],
                joint_mu_overrides={
                    r'^hand1_joint_link_[1-5]_1$': 0.02,
                    r'^hand1_joint_link_[1-5]_2$': 0.02,
                    r'^hand1_joint_link_[1-5]_3$': 0.02,
                }
            )
        else:
            return HandSpec(
                model_format='urdf',
                model_path='psi-oy/InspireHand_OY_Right/InspireHand_OY_Right.urdf',
                mesh_path='psi-oy/InspireHand_OY_Right/meshes',
                contact_points_path=None,
                penetration_points_path=None,
                exclude_links_for_sdf=[],
                joint_mu_overrides={
                    r'^hand2_joint_link_[1-5]_1$': 0.02,
                    r'^hand2_joint_link_[1-5]_2$': 0.02,
                    r'^hand2_joint_link_[1-5]_3$': 0.02,
                }
            )

@dataclass
class EnergyConfig:
    """Energy function weights and thresholds."""
    
    # Energy weights - the "magic" parameters
    w_dis: float = 100.0         # Contact distance weight
    w_pen: float = 125.0         # Penetration penalty weight
    w_spen: float = 10.0         # Self-penetration weight
    w_joints: float = 1.0        # Joint limit weight
    w_vew: float = 0.5           # Wrench ellipse volume weight
    
    # Mode switch: when True, enable uni2bim behavior (per-hand independent optimization)
    use_uni2bim: bool = False
    
    # Energy thresholds for filtering
    thres_fc: float = 0.45        # Force closure threshold
    thres_dis: float = 0.005     # Distance threshold
    thres_pen: float = 0.001     # Penetration threshold
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for energy function calls."""
        return {
            'w_dis': self.w_dis,
            'w_pen': self.w_pen,
            'w_spen': self.w_spen,
            'w_joints': self.w_joints,
            'w_vew': self.w_vew
        }


@dataclass
class OptimizerConfig:
    """Optimization algorithm configuration."""
    
    # Annealing parameters
    switch_possibility: float = 0.5      # Contact point switching probability
    starting_temperature: float = 18     # Initial annealing temperature
    temperature_decay: float = 0.95      # Temperature decay rate
    annealing_period: int = 30           # Annealing period steps
    
    # Step size parameters
    step_size: float = 0.005             # Base step size
    stepsize_period: int = 50            # Step size decay period
    momentum: float = 0.98               # RMSProp momentum parameter
    
    # Iteration settings
    num_iterations: int = 10000           # Number of optimization iterations
    
    # Compatibility properties for MALAOptimizer
    @property
    def initial_temperature(self) -> float:
        """Alias for starting_temperature (MALA compatibility)."""
        return self.starting_temperature
    
    @property
    def cooling_schedule(self) -> float:
        """Alias for temperature_decay (MALA compatibility)."""
        return self.temperature_decay
        
    @property
    def preconditioning_decay(self) -> float:
        """Alias for momentum (MALA compatibility)."""
        return self.momentum
    
    @property 
    def langevin_noise_factor(self) -> float:
        """Langevin noise factor (default for MALA)."""
        return 0.1


@dataclass
class InitializationConfig:
    """Hand initialization parameters."""
    
    # Spatial constraints
    distance_lower: float = 0.2          # Minimum initial distance from object
    distance_upper: float = 0.3          # Maximum initial distance from object
    theta_lower: float = -math.pi / 6    # Minimum rotation angle
    theta_upper: float = math.pi / 6     # Maximum rotation angle
    
    # Joint initialization
    jitter_strength: float = 0.1         # Joint angle randomization strength
    # Joint init bias: mu = lower + bias*(upper-lower); set small to open hand
    joint_mu_bias: float = 0.05
    joint_mu_mode: str = 'bias'           # 'bias' | 'mid' | 'zero'
    joint_mu_overrides_left: Dict[str, float] = field(default_factory=dict)
    joint_mu_overrides_right: Dict[str, float] = field(default_factory=dict)
    
    # Contact points
    num_contacts: int = 4                # Number of contact points per hand

    # Target-based initialization (optional)
    init_at_targets: bool = False        # Enable target-based init
    left_target: Optional[str] = None    # "x y z" in object coords
    right_target: Optional[str] = None   # "x y z" in object coords
    target_distance: float = 0.22        # Back-off distance from target along approach dir
    target_jitter_dist: float = 0.0      # +/- distance jitter
    target_jitter_angle: float = 0.0     # cone half-angle (radians) around approach dir
    target_twist_range: float = 0.0      # random twist around approach axis in [-range, +range] (radians)
    twist_mirror: bool = False           # if True, right hand uses -phi instead of phi
    right_twist_offset: float = 0.0      # constant additional offset added to right-hand twist
    # Interactive selection
    interact: bool = False               # launch interactive UI to select A/B
    snap_to_surface: bool = False        # snap sphere to nearest surface
    ui_port: int = 8050                  # dash server port (if used)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    # Hand model parameters
    n_surface_points: int = 2000         # Number of surface points for sampling
    object_scale_multiplier: float = 1.0 # Global multiplier applied to object scale
    
    # Object model parameters  
    num_samples: int = 2000              # Number of object surface samples
    size: str = 'large'                  # Object size setting
    
    # Batch processing
    batch_size: int = 128                # Batch size for single experiments
    batch_size_each: int = 5             # Batch size per object for large-scale
    max_total_batch_size: int = 100      # Maximum total batch size for multi-GPU


@dataclass
class VisConfig:
    """Visualization/recording configuration for optimization video."""
    enabled: bool = False
    sample_object_index: int = 0
    sample_local_index: int = 0
    record_num: int = 1
    frame_stride: int = 50
    fps: int = 30
    width: int = 900
    height: int = 900
    out_dirname: str = 'frames'
    video_filename: str = 'optimization.mp4'
    show_contacts: bool = False
    bg_color: str = '#E2F0D9'


@dataclass
class MetricsConfig:
    """Minimal metrics recording configuration."""
    enabled: bool = False
    stride: int = 50


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Basic settings
    name: str = 'exp_2025'               # Experiment name
    seed: int = 1                        # Random seed
    gpu: str = "0"                       # GPU device ID
    vis_init: bool = False               # Reduce initial hand distance for visualization
    
    # Object selection
    object_code_list: List[str] = field(default_factory=lambda: [
        'Cole_Hardware_Dishtowel_Multicolors',
        'Curver_Storage_Bin_Black_Small',
        'Hasbro_Monopoly_Hotels_Game',
        'Breyer_Horse_Of_The_Year_2015',
        'Schleich_S_Bayala_Unicorn_70432'
    ])
    
    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vis: VisConfig = field(default_factory=VisConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    # Hand specs
    left_hand: HandSpec = field(default_factory=lambda: HandSpec.preset_shadow('left'))
    right_hand: HandSpec = field(default_factory=lambda: HandSpec.preset_shadow('right'))
    
    # Derived properties
    @property
    def total_batch_size(self) -> int:
        return len(self.object_code_list) * self.model.batch_size
    
    @property
    def device_str(self) -> str:
        return f"cuda:{self.gpu}" if self.gpu != "cpu" else "cpu"
    
    def update_from_args(self, args) -> None:
        """Update configuration from argparse Namespace."""
        # Update basic settings
        for attr in ['name', 'seed', 'gpu']:
            if hasattr(args, attr):
                setattr(self, attr, getattr(args, attr))
        if hasattr(args, 'vis_init'):
            self.vis_init = bool(getattr(args, 'vis_init'))
        
        # Update object list
        if hasattr(args, 'object_code_list'):
            self.object_code_list = args.object_code_list
            
        # Update energy weights
        energy_attrs = ['w_dis', 'w_pen', 'w_spen', 'w_joints']
        for attr in energy_attrs:
            if hasattr(args, attr):
                setattr(self.energy, attr, getattr(args, attr))
                
        # Update optimizer parameters
        opt_attrs = ['switch_possibility', 'starting_temperature', 'temperature_decay', 
                    'annealing_period', 'step_size', 'stepsize_period', 'momentum', 'num_iterations']
        for attr in opt_attrs:
            if hasattr(args, attr):
                setattr(self.optimizer, attr, getattr(args, attr))
                
        # Update initialization parameters (only override when CLI provides a non-None value)
        init_attrs = ['distance_lower', 'distance_upper', 'theta_lower', 'theta_upper', 
                     'jitter_strength', 'joint_mu_bias', 'joint_mu_mode', 'num_contacts', 'init_at_targets', 'left_target',
                     'right_target', 'target_distance', 'target_jitter_dist', 'target_jitter_angle',
                     'target_twist_range', 'twist_mirror', 'right_twist_offset', 'interact', 'snap_to_surface', 'ui_port']
        for attr in init_attrs:
            if hasattr(args, attr):
                val = getattr(args, attr)
                if val is not None:
                    setattr(self.initialization, attr, val)
                
        # Update model parameters
        model_attrs = ['batch_size', 'batch_size_each', 'max_total_batch_size', 'object_scale_multiplier']
        for attr in model_attrs:
            if hasattr(args, attr):
                setattr(self.model, attr, getattr(args, attr))
                
        # Update energy thresholds
        thresh_attrs = ['thres_fc', 'thres_dis', 'thres_pen']
        for attr in thresh_attrs:
            if hasattr(args, attr):
                setattr(self.energy, attr, getattr(args, attr))

        # Mode switch mapping (default behavior unchanged unless explicitly set)
        if hasattr(args, 'mode'):
            mode_val = getattr(args, 'mode')
            if isinstance(mode_val, str) and mode_val.lower() == 'uni2bim':
                self.energy.use_uni2bim = True
            else:
                # Keep default False for any other value (including 'default')
                self.energy.use_uni2bim = False

        # Update visualization parameters (if provided)
        vis_map = {
            'vis': 'enabled',
            'vis_frame_stride': 'frame_stride',
            'vis_obj': 'sample_object_index',
            'vis_local': 'sample_local_index',
            'vis_record_num': 'record_num',
            'vis_fps': 'fps',
            'vis_width': 'width',
            'vis_height': 'height',
            'vis_contacts': 'show_contacts'
        }
        for cli_name, field_name in vis_map.items():
            if hasattr(args, cli_name):
                setattr(self.vis, field_name, getattr(args, cli_name))

        # Hand selection (simple presets)
        # Global hand selector overrides both sides unless side-specific provided
        hand_choice = getattr(args, 'hand', None) if hasattr(args, 'hand') else None
        left_choice = getattr(args, 'left_hand', None) if hasattr(args, 'left_hand') else None
        right_choice = getattr(args, 'right_hand', None) if hasattr(args, 'right_hand') else None

        def apply_choice(choice: Optional[str], side: str):
            if choice is None:
                return
            choice_l = str(choice).lower()
            if choice_l in ['shadow', 'shadowhand', 'mjcf']:
                spec = HandSpec.preset_shadow('left' if side == 'left' else 'right')
            elif choice_l in ['psi_oy', 'psi-oy', 'psioy', 'urdf']:
                spec = HandSpec.preset_psi_oy('left' if side == 'left' else 'right')
            else:
                # Unknown keyword, ignore
                return
            if side == 'left':
                self.left_hand = spec
            else:
                self.right_hand = spec

        # Apply global first, then side-specific to override
        if hand_choice is not None:
            apply_choice(hand_choice, 'left')
            apply_choice(hand_choice, 'right')
        if left_choice is not None:
            apply_choice(left_choice, 'left')
        if right_choice is not None:
            apply_choice(right_choice, 'right')

        # Metrics CLI mapping
        if hasattr(args, 'metrics'):
            self.metrics.enabled = getattr(args, 'metrics')
        if hasattr(args, 'metrics_stride') and getattr(args, 'metrics_stride') is not None:
            self.metrics.stride = int(getattr(args, 'metrics_stride'))


DEFAULT_CONFIG = ExperimentConfig()


def get_config(config_override: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """
    Get configuration with optional overrides.
    
    Args:
        config_override: Dictionary of configuration overrides
        
    Returns:
        ExperimentConfig: Configuration instance
    """
    config = ExperimentConfig()
    
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Try to set nested attributes
                parts = key.split('.')
                if len(parts) == 2 and hasattr(config, parts[0]):
                    sub_config = getattr(config, parts[0])
                    if hasattr(sub_config, parts[1]):
                        setattr(sub_config, parts[1], value)
                        
    return config


def create_config_from_args(args) -> ExperimentConfig:
    """
    Create configuration from command line arguments.
    
    Args:
        args: argparse.Namespace from command line arguments
        
    Returns:
        ExperimentConfig: Configuration instance
    """
    config = ExperimentConfig()
    config.update_from_args(args)
    return config