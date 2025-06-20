from dataclasses import dataclass, field
from typing import List, Optional

# Data configuration
@dataclass
class DataConfig:
    """
    Configuration for dataset loading and metadata.
    Controls which dataset to use, its location, file name,
    target feature, and data frequency for time encoding.
    """
    name: str = 'ETTh1'  # Dataset type
    root_path: str = './data/ETT/'  # Root path of the data file
    data_path: str = 'ETTh1.csv'  # Data file name
    features: str = 'M'  # Forecasting task: M (multivariate), S (univariate), MS (multivariate to univariate)
    target: str = 'OT'  # Target feature in S or MS task
    freq: str = 'h'  # Frequency for time features encoding
    checkpoints: str = './checkpoints/'  # Location to save model checkpoints
    seasonal_patterns: str = 'Monthly'  # Subset for M4 dataset

# Forecasting configuration
@dataclass
class ForecastConfig:
    """
    Configuration for forecasting sequence setup.
    Defines input, label, and prediction lengths and whether
    the output should be inverted to the original scale.
    """
    seq_len: int = 96  # Input sequence length
    label_len: int = 48  # Start token length
    pred_len: int = 96  # Prediction sequence length
    inverse: bool = False  # Whether to inverse output data

# Imputation task configuration
@dataclass
class ImputationConfig:
    """
    Configuration for the imputation task.
    Controls the percentage of data to mask for imputation.
    """
    mask_rate: float = 0.25  # Mask ratio for imputation task

# Anomaly detection configuration
@dataclass
class AnomalyConfig:
    """
    Configuration for the anomaly detection task.
    Sets the assumed anomaly ratio in the dataset.
    """
    anomaly_ratio: float = 0.25  # Prior anomaly ratio

# Model configuration
@dataclass
class ModelConfig:
    """
    Core model configuration including architectural
    parameters like attention, layers, activation, and
    downsampling options.
    """
    expand: int = 2  # Expansion factor for Mamba model
    d_conv: int = 4  # Convolution kernel size for Mamba model
    top_k: int = 5  # Top k for TimesBlock
    num_kernels: int = 6  # Number of kernels for Inception module
    enc_in: int = 7  # Encoder input size
    dec_in: int = 7  # Decoder input size
    c_out: int = 7  # Output size
    d_model: int = 512  # Model dimension
    n_heads: int = 8  # Number of attention heads
    e_layers: int = 2  # Number of encoder layers
    d_layers: int = 1  # Number of decoder layers
    d_ff: int = 2048  # Dimension of feedforward network
    moving_avg: int = 25  # Window size of moving average
    factor: int = 1  # Attention factor
    distil: bool = True  # Use distilling in encoder
    dropout: float = 0.1  # Dropout rate
    embed: str = 'timeF'  # Time features encoding method
    activation: str = 'gelu'  # Activation function
    channel_independence: int = 1  # Channel independence for FreTS model
    decomp_method: str = 'moving_avg'  # Series decomposition method
    use_norm: int = 1  # Whether to use normalization
    down_sampling_layers: int = 0  # Number of down sampling layers
    down_sampling_window: int = 1  # Down sampling window size
    down_sampling_method: Optional[str] = None  # Down sampling method (avg, max, conv)
    seg_len: int = 96  # Segment length for SegRNN
    patch_len: int = 16  # Patch length for TimeXer

# Optimization configuration
@dataclass
class OptimizationConfig:
    """
    Training configuration for optimization routines,
    early stopping, mixed precision training, and batch processing.
    """
    num_workers: int = 10  # Number of data loader workers
    itr: int = 1  # Number of experiment repetitions
    train_epochs: int = 10  # Number of training epochs
    batch_size: int = 32  # Batch size for training
    patience: int = 3  # Early stopping patience
    learning_rate: float = 0.0001  # Learning rate for optimizer
    loss: str = 'MSE'  # Loss function
    lradj: str = 'type1'  # Learning rate adjustment type
    use_amp: bool = False  # Use automatic mixed precision training

# GPU configuration
@dataclass
class GPUConfig:
    """
    Hardware configuration to manage GPU usage and
    multi-GPU setups.
    """
    use_gpu: bool = True  # Whether to use GPU
    gpu: int = 0  # GPU index
    gpu_type: str = 'cuda'  # GPU type: cuda or mps
    use_multi_gpu: bool = False  # Whether to use multiple GPUs
    devices: str = '0,1,2,3'  # Multi-GPU device IDs

# Projector configuration
@dataclass
class ProjectorConfig:
    """
    Configuration for de-stationary projector architecture.
    Controls hidden layer sizes and depth.
    """
    p_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])  # Projector hidden layer dimensions
    p_hidden_layers: int = 2  # Number of hidden layers in projector

# Metrics configuration
@dataclass
class MetricsConfig:
    """
    Configuration for evaluation metrics such as DTW.
    Note: DTW is computationally expensive.
    """
    use_dtw: bool = False  # Whether to use DTW metric

# Augmentation configuration
@dataclass
class AugmentationConfig:
    """
    Data augmentation configuration including a variety of
    preset transformations such as jitter, scaling, permutation,
    and DTW-based warping techniques.
    """
    augmentation_ratio: int = 0  # Number of times to augment
    seed: int = 2  # Randomization seed
    jitter: bool = False  # Jitter augmentation
    scaling: bool = False  # Scaling augmentation
    permutation: bool = False  # Equal length permutation augmentation
    randompermutation: bool = False  # Random length permutation augmentation
    magwarp: bool = False  # Magnitude warp augmentation
    timewarp: bool = False  # Time warp augmentation
    windowslice: bool = False  # Window slice augmentation
    windowwarp: bool = False  # Window warp augmentation
    rotation: bool = False  # Rotation augmentation
    spawner: bool = False  # SPAWNER augmentation
    dtwwarp: bool = False  # DTW warp augmentation
    shapedtwwarp: bool = False  # Shape DTW warp augmentation
    wdba: bool = False  # Weighted DBA augmentation
    discdtw: bool = False  # Discriminative DTW warp augmentation
    discsdtw: bool = False  # Discriminative shape DTW warp augmentation
    extra_tag: str = ''  # Extra tag for experiments

# Overall experiment configuration
@dataclass
class ExperimentConfig:
    """
    Root configuration class encapsulating all experiment
    settings, including task type, model identity, and links
    to all specialized configurations.
    """
    task_name: str = 'long_term_forecast'  # Task type
    is_training: int = 1  # Training status: 1 for training, 0 for testing
    model_id: str = 'test'  # Model ID
    model: str = 'Autoformer'  # Model name
    des: str = 'test'  # Experiment description
    experiment_id: str = "test" # Experiment unique id

    data: DataConfig = field(default_factory=DataConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    anomaly_detection: AnomalyConfig = field(default_factory=AnomalyConfig)
    model_params: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

class FlatExperimentConfig:
    def __init__(self, config: ExperimentConfig):
        self._config = config

    def __getattr__(self, name):
        if hasattr(self._config, name):
            return getattr(self._config, name)
        if hasattr(self._config.data, name):
            return getattr(self._config.data, name)
        if hasattr(self._config.forecast, name):
            return getattr(self._config.forecast, name)
        if hasattr(self._config.model_params, name):
            return getattr(self._config.model_params, name)
        if hasattr(self._config.augmentation, name):
            return getattr(self._config.augmentation, name)
        if hasattr(self._config.optimization, name):
            return getattr(self._config.optimization, name)
        raise AttributeError(f"'{type(self._config).__name__}' object has no attribute '{name}'")