import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_scheduler
import wandb
from prepare.process import AIDS_GraphDataset
from models.gnn import GNN
from trainer import Trainer
from config import Config
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from utils import expand_dataset

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    cfg = instantiate(cfg)

    full_dataset = AIDS_GraphDataset(cfg.dataset.path)

    split_idx = int(0.8 * len(full_dataset))
    train_data = expand_dataset(full_dataset[:split_idx])
    test_data = expand_dataset(full_dataset[split_idx:])

    train_loader = GeoDataLoader(
        train_data,
        batch_size=cfg.dataloader.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory
    )
    test_loader = GeoDataLoader(
        test_data,
        batch_size=cfg.dataloader.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory
    )

    gnn = GNN(
        in_channels=full_dataset[0].x.size(1),
        edge_attr_dim=full_dataset[0].edge_attr.size(1),
        hidden_channels=cfg.gnn.hidden_dim,
        out_channels=cfg.gnn.out_channels,
        dropout=cfg.gnn.dropout,
        heads=cfg.gnn.attn_heads,
        pretrained_weights_dir=cfg.gnn.pretrained_weights_dir,
    ).to(device)

    tokenizer = None
    llm = None
    projector = None

    if cfg.llm.use_llm:
        model_name = cfg.llm.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.model_max_length = 1024
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            'additional_special_tokens': ['<|GRAPH_START|>', '<|GRAPH_EMBEDDING|>', '<|GRAPH_END|>']
        })

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
        )

        llm = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
        llm.resize_token_embeddings(len(tokenizer))

        if cfg.llm.train:
            lora_config = LoraConfig(
                r=cfg.llm.lora_rank,
                lora_alpha=cfg.llm.lora_alpha,
                target_modules=OmegaConf.to_container(cfg.llm.lora_target_modules, resolve=True),
                lora_dropout=cfg.llm.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            llm = get_peft_model(llm, lora_config)
        else:
            for param in llm.parameters():
                param.requires_grad = False

        projector = torch.nn.Linear(cfg.gnn.out_channels, llm.config.hidden_size).to(device)
    else:
        projector = torch.nn.Linear(cfg.gnn.out_channels, 1).to(device)

    if cfg.projector.pretrained_weights_dir:
        projector.load_state_dict(
            torch.load(cfg.projector.pretrained_weights_dir, map_location=device)
        )

    optimizer_params = []
    if cfg.gnn.train:
        optimizer_params.append({"params": gnn.parameters(), "lr": cfg.gnn.lr})
    if cfg.projector.train:
        optimizer_params.append({"params": projector.parameters(), "lr": cfg.projector.lr})
    if cfg.llm.use_llm and cfg.llm.train:
        optimizer_params.append({"params": llm.parameters(), "lr": cfg.llm.lr})

    optimizer = torch.optim.Adam(optimizer_params)

    scheduler = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=len(train_loader) * 10,
    )

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)

    trainer = Trainer(
        gnn=gnn,
        llm=llm,
        projector=projector,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        device=device,
        config=cfg,
    )

    trainer.train()

if __name__ == "__main__":
    main()
