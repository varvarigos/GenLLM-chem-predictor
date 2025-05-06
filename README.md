markdown
# GNN–LLM for Molecular Descriptor Prediction

This repository implements a hybrid architecture that combines Graph Neural Networks (GNNs) with a Large Language Model (LLM) to generate molecular property descriptors from graph-structured molecular data. It supports both training and inference, as well as a simple UI for guided predictions.

## 🔧 Setup

To install the required environment and dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
````

## 🚀 Running the Code

To train and run validation with the hybrid GNN–LLM pipeline, execute:

```bash
python3 main.py
```

This will load the configured dataset, fine-tune the LLM using LoRA, and produce predictions for molecular descriptors based on graph inputs.

## 🧪 User Interface (Optional)

An interface for testing the model is provided, created using Gradio:

```bash
python3 UI/interface.py
```

**Note:**

* The interface currently supports **task selection and graph upload** via `.gxf` files.
* **Free-form Q\&A is not yet supported in the UI**, but the underlying fine-tuned model can be used independently for open-ended questions about molecular graphs.

## 📦 Dependencies

Key Python packages (see `requirements.txt`) include:

* `transformers`
* `torch`
* `networkx`
* `rdkit`
* `datasets`
* `peft`
* `tqdm`

Install them all with:

```bash
pip install -r requirements.txt
```

## 📁 Datasets and Models

* **Dataset**: We use the **AIDS Antiviral Screen Dataset**, where each molecule is represented as a graph with one-hot encoded atom and bond types.
* **Descriptors**: Computed via [`RDKit`](https://www.rdkit.org/) using SMILES strings translated from the graph.
* **Model**: The LLM used is [`microsoft/phi-1_5`](https://huggingface.co/microsoft/phi-1_5), fine-tuned using LoRA. Graph embeddings are injected into the LLM’s prompt stream using special tokens.
