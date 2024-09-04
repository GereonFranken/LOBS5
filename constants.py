from dataclasses import dataclass
from typing import Optional

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
@dataclass(frozen=True)
class TrainArgs(metaclass=Singleton):
    USE_WANDB: bool = True
    wandb_project: str = "LOBMamba"
    wandb_entity: str = "gereon-franken-oxford"
    dir_name: str = "/Users/gereonfranken/Projects/phd/LOBS5/data/minimal_test"
    dataset: str = "lobster-prediction"
    masking: str = "causal"
    use_book_data: bool = True
    use_simple_book: bool = False
    book_transform: bool = True
    book_depth: int = 500
    restore: Optional[str] = None
    restore_step: Optional[int] = None
    msg_seq_len: int = 10
    n_data_workers: int = 2
    n_message_layers: int = 2
    n_book_pre_layers: int = 1
    n_book_post_layers: int = 1
    n_layers: int = 1
    d_model: int = 16
    ssm_size_base: int = 32
    blocks: int = 2
    expand_factor: int = 2
    C_init: str = "trunc_standard_normal"
    discretization: str = "zoh"
    mode: str = "pool"
    activation_fn: str = "half_glu1"
    conj_sym: bool = False
    clip_eigs: bool = True
    bidirectional: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    prenorm: bool = True
    batchnorm: bool = True
    bn_momentum: float = 0.95
    bsz: int = 2
    num_devices: int = 1
    epochs: int = 10
    early_stop_patience: int = 1000
    ssm_lr_base: float = 0.0005
    lr_factor: float = 1.0
    dt_global: bool = False
    lr_min: int = 0
    cosine_anneal: bool = True
    warmup_end: int = 1
    lr_patience: int = 1000000
    reduce_factor: float = 1.0
    p_dropout: float = 0.0
    weight_decay: float = 0.05
    opt_config: str = "standard"
    jax_seed: int = 42