from pathlib import Path
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class WandbConfig:
    project: str = "gnn-llm-chem"
    name: str = "gnn-llm-training"

@dataclass
class TrainerConfig:
    epochs: int = 15
    model_dir: Path = Path("./model_weights")

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)

@dataclass
class LRSchedulerConfig:
    name: str = "linear"
    lr: float = 1e-4
    warmup_steps: int = 0

@dataclass
class LLMConfig:
    use_llm: bool = False
    train: bool = True
    model_name: str = "meta-llama/Llama-3.2-1B"
    use_context_prompt: bool = True
    lr: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

@dataclass
class InferenceConfig:
    max_length: int = 256
    max_new_tokens: int = 100
    repetition_penalty: float = 2.5
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

@dataclass
class ProjectorConfig:
    lr: float = 1e-3
    pretrained_weights_dir: Optional[Path] = None
    train: bool = True

    def __post_init__(self):
        if self.pretrained_weights_dir is not None:
            self.pretrained_weights_dir = Path(self.pretrained_weights_dir)

@dataclass
class GNNConfig:
    hidden_dim: int = 64
    dropout: float = 0.2
    out_channels: int = 2048
    attn_heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    pretrained_weights_dir: Optional[Path] = None
    modality_embedding_trainable: bool = True
    train: bool = True
    
    def __post_init__(self):
        if self.pretrained_weights_dir is not None:
            self.pretrained_weights_dir = Path(self.pretrained_weights_dir)

@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_batch_size: int = 72
    val_batch_size: int = 72
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class DatasetConfig:
    path: Path = Path("/home/andreas/DL_CW/AIDS/data")
  
    def __post_init__(self):
        self.path = Path(self.path)

@dataclass
class InterfaceConfig:
    llm_path: Path = Path("/home/andreas/DL_CW/outputs/2025-04-29__00-10-55/model_weights/ep10/llm_model")
    gnn_path: Path = Path("/home/andreas/DL_CW/outputs/2025-04-29__00-10-55/model_weights/ep10/gnn_model")
    projector_path: Path = Path("/home/andreas/DL_CW/outputs/2025-04-29__00-10-55/model_weights/ep10/projector_model")
    modality_emb_path: Path = Path("/home/andreas/DL_CW/outputs/2025-04-29__00-10-55/model_weights/ep10/modality_emb_model")

    def __post_init__(self):
        self.llm_path = Path(self.llm_path)
        self.gnn_path = Path(self.gnn_path)
        self.projector_path = Path(self.projector_path)
        self.modality_emb_path = Path(self.modality_emb_path)

@dataclass
class Config:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    train: TrainerConfig = field(default_factory=TrainerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
