from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceConfig:
    data_dir: str
    processed_dir: str
    model_dir: str
    output_dir: str
    seed: int
    exp_name: str
    device: str
    batch_size: int
    num_workers: int


@dataclass
class PrepareDataConfig:
    data_path: str
    phase: str
    save_path: str

@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    split_dir: str
    output_dir: str
    model_dir: str
    sub_dir: str


@dataclass
class SplitConfig:
    name: str
    train_id: list
    val_id: list


@dataclass
class ModelConfig:
    name: str
    params: dict

@dataclass
class EEGFeatExtractorConfig:
    name: str
    params: dict

@dataclass
class TrainerConfig:
    epochs: int
    device: 1
    accelerator: str
    use_amp: bool
    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: str
    monitor_mode: str
    val_check_interval: int


@dataclass
class OptimizerConfig:
    lr: float


@dataclass
class SchedulerConfig:
    num_warmup_steps: int


@dataclass
class WeightConfig:
    exp_name: str
    run_name: str

@dataclass
class AugConfig:
  do_mixup: bool
  do_cutmix: bool
  do_horizontal_flip: bool
  do_xy_masking: bool 

@dataclass
class TrainConfig:
    exp_name: str
    seed: int
    batch_size: int
    num_workers: int
    num_samples: int
    device: str
    early_stopping_rounds: int
    in_channels: int
    num_classes: int
    split_by: str
    spec_img_size: int
    use_eeg_spec: bool
    use_raw_eeg: bool
    use_overlap: bool
    exclude_difficult_data: bool
    pretrained: bool
    pretrained_exp: str
    labels: list
    n_folds: int
    n_folds_total: 5
    n_folds_start: 0
    fold_ver: int
    loss_ver: int
    data_process_ver: int
    eeg_spec_version: int
    do_label_smoothing: bool
    hard_sample: bool
    pred_confidence: bool
    use_target_pattern: bool
    cut_edge_spec: bool
    cut_spec_width: int
    dir: DirConfig
    model: ModelConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    aug: AugConfig



