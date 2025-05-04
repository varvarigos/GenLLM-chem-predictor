from tqdm import tqdm
import torch
import wandb
import os
from utils import context_metrics


class Trainer:
    def __init__(
        self,
        gnn,
        llm,
        projector,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        tokenizer,
        config,
        device,
    ):
        self.gnn = gnn
        self.llm = llm
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = device
        self.projector = projector


    def train(self):
        self.gnn.train()
        self.projector.train()
        self.llm.train() if self.cfg.llm.train else self.llm.eval()

        for epoch in range(self.cfg.train.epochs):
            total_loss = 0
            for batch in tqdm(self.train_loader):
                batch = batch.to(self.device, non_blocking=True)

                graph_embeddings = self.gnn(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                ).unsqueeze(1).to(torch.float32)
                graph_embeddings = self.projector(graph_embeddings).to(torch.float16)

                if self.cfg.llm.use_context_prompt:
                    context_batch = context_metrics(batch)
                    prompts = [
                        f"<|GRAPH_START|><|GRAPH_EMBEDDING|><|GRAPH_END|>"
                        f"Given this graph representation and the following characteristics of a chemical compound:\n"
                        f"- The compound has {ctx['num_nodes']} atoms and {ctx['num_edges']} bonds.\n"
                        f"- The average atom degree is {ctx['avg_degree']:.2f}.\n"
                        f"- The most common atom type is {ctx['most_common_atom']}.\n"
                        f"- The fraction of single bonds is {ctx['frac_single']:.2f}, "
                        f"double bonds {ctx['frac_double']:.2f}, and triple bonds {ctx['frac_triple']:.2f}.\n"
                        f"Please give me the {desc.lower()} of this chemical compound.\nAnswer:\n"
                        for ctx, desc in zip(context_batch, batch.descriptor)
                    ]
                else:
                    prompts = [
                        f"<|GRAPH_START|><|GRAPH_EMBEDDING|><|GRAPH_END|>"
                        f"Given this graph representation of a chemical compound, please give me its {desc.lower()}.\nAnswer:\n"
                        for desc in batch.descriptor
                    ]

                targets = [f"The {desc.lower()} of the compound is {val.item():.2f}." for desc, val in zip(batch.descriptor, batch.y)]
                full_texts = [p + t for p, t in zip(prompts, targets)]

                all_tokenized = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
                labels = all_tokenized.input_ids.to(self.device)
                attention_mask = (labels != self.tokenizer.pad_token_id).long()

                prompt_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
                input_embeds = self.llm.get_input_embeddings()(prompt_tokenized).to(torch.float16)

                # === Pad input_embeds to match label length ===
                B, L_prompt, D = input_embeds.shape
                L_full = labels.shape[1]
                pad_len = L_full - L_prompt
                if pad_len > 0:
                    pad_embeds = torch.zeros((B, pad_len, D), dtype=input_embeds.dtype, device=input_embeds.device)
                    input_embeds = torch.cat([input_embeds, pad_embeds], dim=1)
                    attention_mask[:, L_prompt:] = 0

                graph_embed_id = self.tokenizer.convert_tokens_to_ids('<|GRAPH_EMBEDDING|>')
                for b in range(input_embeds.size(0)):
                    graph_embed_pos = (prompt_tokenized[b] == graph_embed_id).nonzero(as_tuple=True)[0].item()
                    input_embeds[b, graph_embed_pos, :] = graph_embeddings[b, 0, :]

                position_ids = torch.arange(input_embeds.size(1), device=self.device).unsqueeze(0).expand(input_embeds.size(0), -1)

                outputs = self.llm(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids,
                )

                loss = outputs.loss
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                wandb.log({"train_loss_per_step": loss})

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
            wandb.log({"train_loss_per_epoch": avg_loss, "epoch": epoch})

            if epoch % 1 == 0:
                gnn_dir = os.path.join(self.cfg.train.model_dir, f"ep{epoch}", "gnn_model")
                projector_dir = os.path.join(self.cfg.train.model_dir, f"ep{epoch}", "projector_model")
                llm_dir = os.path.join(self.cfg.train.model_dir, f"ep{epoch}", "llm_model")
                os.makedirs(gnn_dir, exist_ok=True)
                os.makedirs(projector_dir, exist_ok=True)
                if self.cfg.llm.train:
                    os.makedirs(llm_dir, exist_ok=True)

                torch.save(self.gnn.state_dict(), os.path.join(gnn_dir, "model.pth"))
                torch.save(self.projector.state_dict(), os.path.join(projector_dir, "model.pth"))
                if self.cfg.llm.train:
                    self.llm.save_pretrained(llm_dir)
                
                self.test()


    @torch.no_grad()
    def test(self):
        self.gnn.eval()
        self.projector.eval()
        self.llm.eval()

        total_loss = 0
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            batch = batch.to(self.device, non_blocking=True)

            graph_embeddings = self.gnn(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            ).unsqueeze(1).to(torch.float32)
            graph_embeddings = self.projector(graph_embeddings).to(torch.float16)

            if self.cfg.llm.use_context_prompt:
                context_batch = context_metrics(batch)
                prompts = [
                    f"<|GRAPH_START|><|GRAPH_EMBEDDING|><|GRAPH_END|>"
                    f"Given this graph representation and the following characteristics of a chemical compound:\n"
                    f"- The compound has {ctx['num_nodes']} atoms and {ctx['num_edges']} bonds.\n"
                    f"- The average atom degree is {ctx['avg_degree']:.2f}.\n"
                    f"- The most common atom type is {ctx['most_common_atom']}.\n"
                    f"- The fraction of single bonds is {ctx['frac_single']:.2f}, "
                    f"double bonds {ctx['frac_double']:.2f}, and triple bonds {ctx['frac_triple']:.2f}.\n"
                    f"Please give me the {desc.lower()} of this chemical compound.\nAnswer:\n"
                    for ctx, desc in zip(context_batch, batch.descriptor)
                ]
            else:
                prompts = [
                    f"<|GRAPH_START|><|GRAPH_EMBEDDING|><|GRAPH_END|>"
                    f"Given this graph representation of a chemical compound, please give me its {desc.lower()}.\nAnswer:\n"
                    for desc in batch.descriptor
                ]

            targets = [f"The {desc.lower()} of the compound is {val.item():.2f}." for desc, val in zip(batch.descriptor, batch.y)]
            full_texts = [p + t for p, t in zip(prompts, targets)]

            all_tokenized = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
            labels = all_tokenized.input_ids.to(self.device)
            attention_mask = (labels != self.tokenizer.pad_token_id).long()

            prompt_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
            input_embeds = self.llm.get_input_embeddings()(prompt_tokenized).to(torch.float16)

            # === Pad input_embeds to match label length ===
            B, L_prompt, D = input_embeds.shape
            L_full = labels.shape[1]
            pad_len = L_full - L_prompt
            if pad_len > 0:
                pad_embeds = torch.zeros((B, pad_len, D), dtype=input_embeds.dtype, device=input_embeds.device)
                input_embeds = torch.cat([input_embeds, pad_embeds], dim=1)
                attention_mask[:, L_prompt:] = 0

            graph_embed_id = self.tokenizer.convert_tokens_to_ids('<|GRAPH_EMBEDDING|>')
            for b in range(input_embeds.size(0)):
                graph_embed_pos = (prompt_tokenized[b] == graph_embed_id).nonzero(as_tuple=True)[0].item()
                input_embeds[b, graph_embed_pos, :] = graph_embeddings[b, 0, :]

            position_ids = torch.arange(input_embeds.size(1), device=self.device).unsqueeze(0).expand(input_embeds.size(0), -1)

            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
            )

            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        wandb.log({"test_loss": avg_loss})

        self.gnn.train()
        self.projector.train()
        self.llm.train() if self.cfg.llm.train else self.llm.eval()

        return avg_loss
