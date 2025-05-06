from tqdm import tqdm
import torch
import wandb
import os
from utils import context_metrics
import random
import re
import torch.nn.functional as F

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

        self.target_templates = [
            "The {desc} of the compound is {val:.2f}",
            "This compound has a {desc} of {val:.2f}",
            "Predicted {desc}: {val:.2f}",
            "Its {desc} is approximately {val:.2f}",
            "Measured {desc} value: {val:.2f}",
            "The {desc} value is {val:.2f}",
            "The compound's {desc} is {val:.2f}",
            "The {desc} of this compound is {val:.2f}",
            "{desc} = {val:.2f}.",
            "According to the model, the {desc} is {val:.2f}",
            "This molecule exhibits a {desc} of {val:.2f}",
            "LLM prediction for {desc}: {val:.2f}",
            "For this chemical, the {desc} equals {val:.2f}",
            "A reasonable estimate for {desc} is {val:.2f}",
            "Computed value for {desc}: {val:.2f}",
        ]

    def train(self):
        self.gnn.train()
        self.projector.train()
        
        if self.cfg.llm.use_llm:
            if self.cfg.llm.train:
                self.llm.train()
            else:
                self.llm.eval()

        for epoch in range(self.cfg.train.epochs):
            total_loss = 0
            for batch in tqdm(self.train_loader):
                batch = batch.to(self.device, non_blocking=True)

                # === GNN Embedding ===
                graph_embeddings = self.gnn(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                ).unsqueeze(1).to(torch.float32)

                if not self.cfg.llm.use_llm:
                    preds = self.projector(graph_embeddings).squeeze(-1)
                    loss = F.mse_loss(preds, batch.y)
                else:
                    graph_embeddings = self.projector(graph_embeddings).to(torch.float16)

                    # === Prompt Construction ===
                    if self.cfg.llm.use_context_prompt:
                        context_batch = context_metrics(batch)
                        prompts = [
                            f"{self.tokenizer.bos_token}<|GRAPH_START|><|GRAPH_EMBEDDING|><|GRAPH_END|>"
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
                            f"{self.tokenizer.bos_token}<|GRAPH_START|><|GRAPH_EMBEDDING|><|GRAPH_END|>"
                            f"Given this graph representation of a chemical compound, please give me its {desc.lower()}.\nAnswer:\n"
                            for desc in batch.descriptor
                        ]

                    targets = [
                        random.choice(self.target_templates).format(desc=desc.lower(), val=val.item())
                        for desc, val in zip(batch.descriptor, batch.y)
                    ]

                    # Tokenize prompts and targets
                    prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    inputs = self.tokenizer([pp + tt for pp, tt in zip(prompts, targets)], return_tensors="pt", padding=True, truncation=True).to(self.device)
                    padded_prompts_inputs = F.pad(prompt_inputs.input_ids, (0, inputs.input_ids.shape[1] - prompt_inputs.input_ids.shape[1]), value=self.tokenizer.pad_token_id)

                    target_ids = inputs.input_ids.clone()
                    target_ids[torch.logical_and(target_ids != self.tokenizer.pad_token_id, padded_prompts_inputs==target_ids)] = -100

                    input_ids = inputs.input_ids.to(self.device)
                    attention_mask = inputs.attention_mask.to(self.device)

                    # Obtain input embeddings for prompts
                    input_embeds = self.llm.get_input_embeddings()(input_ids).to(torch.float16)

                    # Replace <|GRAPH_EMBEDDING|> token embeddings with actual graph embeddings
                    graph_embed_id = self.tokenizer.convert_tokens_to_ids('<|GRAPH_EMBEDDING|>')
                    for b in range(input_embeds.size(0)):
                        graph_embed_pos = (input_ids[b] == graph_embed_id).nonzero(as_tuple=True)[0].item()
                        input_embeds[b, graph_embed_pos, :] = graph_embeddings[b, 0, :]

                    # Generate position IDs
                    position_ids = torch.arange(input_embeds.size(1), device=self.device).unsqueeze(0).expand(input_embeds.size(0), -1)

                    # Forward pass through the model
                    outputs = self.llm(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        labels=target_ids,
                        position_ids=position_ids,
                        use_cache=False
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
                if self.cfg.llm.use_llm:
                    if self.cfg.llm.train:
                        self.llm.save_pretrained(llm_dir)

                self.test()


    @torch.no_grad()
    def test(self):
        self.gnn.eval()
        self.projector.eval()
        if self.cfg.llm.use_llm:
            self.llm.eval()

        total_loss = 0
        mse_sum = 0
        mae_sum = 0
        valid_preds = 0
        total_preds = 0

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            batch = batch.to(self.device, non_blocking=True)

            graph_embeddings = self.gnn(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            ).unsqueeze(1).to(torch.float32)

            if not self.cfg.llm.use_llm:
                preds = self.projector(graph_embeddings).squeeze(-1)
                loss = F.mse_loss(preds, batch.y)
                total_loss += loss.item()

                for pred_val, true_val in zip(preds.view(-1).tolist(), batch.y.view(-1).tolist()):
                    error = pred_val - true_val
                    mse_sum += error ** 2
                    mae_sum += abs(error)
                    valid_preds += 1
                    total_preds += 1

                continue

            graph_embeddings = self.projector(graph_embeddings).to(torch.float16)

            # Build prompts
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

            # Ground truth (for loss)
            targets = [
                random.choice(self.target_templates).format(desc=desc.lower(), val=val.item())
                for desc, val in zip(batch.descriptor, batch.y)
            ]

            prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            inputs = self.tokenizer([pp + tt for pp, tt in zip(prompts, targets)], return_tensors="pt", padding=True, truncation=True).to(self.device)
            padded_prompts_inputs = F.pad(prompt_inputs.input_ids, (0, inputs.input_ids.shape[1] - prompt_inputs.input_ids.shape[1]), value=self.tokenizer.pad_token_id)

            target_ids = inputs.input_ids.clone()
            target_ids[torch.logical_and(target_ids != self.tokenizer.pad_token_id, padded_prompts_inputs==target_ids)] = -100

            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            input_embeds = self.llm.get_input_embeddings()(input_ids).to(torch.float16)
            graph_embed_id = self.tokenizer.convert_tokens_to_ids('<|GRAPH_EMBEDDING|>')
            for b in range(input_embeds.size(0)):
                graph_embed_pos = (input_ids[b] == graph_embed_id).nonzero(as_tuple=True)[0].item()
                input_embeds[b, graph_embed_pos, :] = graph_embeddings[b, 0, :]

            position_ids = torch.arange(input_embeds.size(1), device=self.device).unsqueeze(0).expand(input_embeds.size(0), -1)

            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=target_ids,
                position_ids=position_ids,
                use_cache=False
            )

            total_loss += outputs.loss.item()

            # Generate predictions
            generated_ids = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=self.cfg.inference.max_new_tokens,
                repetition_penalty=self.cfg.inference.repetition_penalty,
                temperature=self.cfg.inference.temperature,
                top_p=self.cfg.inference.top_p,
                top_k=self.cfg.inference.top_k,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=True,
            )

            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for pred_str, true_val in zip(generated_texts, batch.y):
                total_preds += 1
                match = re.search(r"\b[-+]?[0-9]*\.?[0-9]+\b", pred_str)
                if match:
                    pred_val = float(match.group())
                    true_val = true_val.item()

                    error = pred_val - true_val
                    mse_sum += error ** 2
                    mae_sum += abs(error)
                    valid_preds += 1

        avg_loss = total_loss / len(self.test_loader)
        mse = mse_sum / valid_preds if valid_preds > 0 else float('nan')
        mae = mae_sum / valid_preds if valid_preds > 0 else float('nan')
        valid_pct = 100.0 * valid_preds / total_preds if total_preds > 0 else 0.0

        print(f"Test Loss: {avg_loss:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f} | % Valid: {valid_pct:.2f}%")
        wandb.log({
            "test_loss": avg_loss,
            "test_mse": mse,
            "test_mae": mae,
            "test_valid_pct": valid_pct
        })

        self.gnn.train()
        self.projector.train()
        if self.cfg.llm.use_llm:
            if self.cfg.llm.train:
                self.llm.train()

        return avg_loss, mse, mae, valid_pct
