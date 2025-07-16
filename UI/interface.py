import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import hydra
import gradio as gr
from config import Config
from peft import PeftModel
from models.gnn import GNN
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import context_metrics
from prepare.process import parse_gxl_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Descriptor options
descriptor_names = [
    "Molecular weight", "Lipophilicity (LogP)", "Number of hydrogen donors", "Number of hydrogen acceptors",
    "Heavy atom count", "Topological polar surface area (TPSA)", "Fraction of sp3-hybridized carbons",
    "Number of rings", "Balaban's J index", "Molar refractivity", "Bertz complexity index",
    "NH or OH count", "NO count", "Number of aliphatic rings", "Number of aromatic rings",
    "Number of saturated rings", "Number of heteroatoms", "Number of rotatable bonds",
    "Valence electron count", "Labute ASA"
]

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Config):
    cfg = instantiate(cfg)

    # Load GNN
    gnn = GNN(
        in_channels=31,
        edge_attr_dim=3,
        hidden_channels=cfg.gnn.hidden_dim,
        out_channels=cfg.gnn.out_channels,
        dropout=cfg.gnn.dropout,
        pretrained_weights_dir=os.path.join(cfg.interface.gnn_path, "model.pth"),
        heads=cfg.gnn.attn_heads,
    ).to(device)
    gnn.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model_name)
    tokenizer.model_max_length = 1024
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    # Add special tokens for graph start and end
    special_tokens_dict = {'additional_special_tokens': ['<|GRAPH_START|>', '<|GRAPH_EMBEDDING|>', '<|GRAPH_END|>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Load LLM
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llm.resize_token_embeddings(len(tokenizer))
    if cfg.interface.llm_path:
        llm = PeftModel.from_pretrained(llm, cfg.interface.llm_path).to(device)
    llm.eval()

    # Load Projector
    projector = torch.nn.Linear(cfg.gnn.out_channels, llm.config.hidden_size).to(device)
    projector.load_state_dict(
        torch.load(os.path.join(cfg.interface.projector_path, "model.pth"), map_location=device)
    )
    projector.eval()

    def predict(gxl_file, task_name):
        data = parse_gxl_file(gxl_file.name)
        data = data.to(device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        graph_embeddings = gnn(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch=data.batch
        ).unsqueeze(0).to(torch.float32)
        graph_embeddings = projector(graph_embeddings).to(torch.float16)

        if cfg.llm.use_context_prompt:
            context_list = context_metrics(data)
            context = context_list[0]
            prompt = (
                f"<|GRAPH_START|><|GRAPH_EMBEDDING|><|GRAPH_END|>"
                f"Given this graph representation and the following characteristics of a chemical compound:\n"
                f"- The compound has {context['num_nodes']} atoms and {context['num_edges']} bonds.\n"
                f"- The average atom degree is {context['avg_degree']:.2f}.\n"
                f"- The most common atom type is {context['most_common_atom']}.\n"
                f"- The fraction of single bonds is {context['frac_single']:.2f}, "
                f"double bonds {context['frac_double']:.2f}, and triple bonds {context['frac_triple']:.2f}.\n"
                f"Please give me the {task_name.lower()} of this chemical compound.\nAnswer:\n"
            )
        else:
            prompt = (
                f"{tokenizer.bos_token}<|GRAPH_START|><|GRAPH_EMBEDDING|><|GRAPH_END|>"
                f"Given this graph representation of a chemical compound, please give me its {task_name.lower()}.\nAnswer:\n"
            )

        input_ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt}
            ],
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(device)

        input_embeds = llm.get_input_embeddings()(input_ids).to(torch.float16)

        graph_embed_id = tokenizer.convert_tokens_to_ids('<|GRAPH_EMBEDDING|>')
        graph_embed_pos = (input_ids == graph_embed_id).nonzero(as_tuple=True)[1].item()
        input_embeds[0, graph_embed_pos, :] = graph_embeddings[0, 0, :]

        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device, dtype=torch.long)
        position_ids = torch.arange(input_embeds.size(1), device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                max_new_tokens=cfg.inference.max_new_tokens,
                repetition_penalty=cfg.inference.repetition_penalty,
                temperature=cfg.inference.temperature,
                top_p=cfg.inference.top_p,
                top_k=cfg.inference.top_k,
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            descriptor_idx = descriptor_names.index(task_name)
            real_value = data.y[descriptor_idx].item()
            print(f"\U0001F50E Real value for '{task_name}': {real_value:.2f}")
        except Exception as e:
            print(f"Error finding real label: {e}")

        return output_text


    # === Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown("# 🧬 GenLLM Chemistry Predictor")

        with gr.Row():
            gxl_upload = gr.File(label="Upload GXL Graph File", file_types=[".gxl"])
            task_dropdown = gr.Dropdown(choices=descriptor_names, label="Select Chemical Property")
            predict_button = gr.Button("Predict")

        output_text = gr.Textbox(label="Result", lines=5)

        predict_button.click(
            predict,
            inputs=[gxl_upload, task_dropdown],
            outputs=output_text
        )

    demo.launch(server_port=7880)

if __name__ == "__main__":
    main()
