# GNN–LLM for Molecular Descriptor Prediction

This repository implements a hybrid architecture that combines Graph Neural Networks (GNNs) with a Large Language Model (LLM) to generate molecular property descriptors from graph-structured molecular data. It supports both training and inference, as well as a simple UI for guided predictions.

## 🔧 Setup

Ensure you have Python 3.11 installed before proceeding with the setup. To install the required environment and dependencies:

```bash
# Clone the repository
git clone https://github.com/varvarigos/GenLLM-chem-predictor.git
cd ./GenLLM-chem-predictor

# Create a virtual environment
python3.11 -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Generate token for Hugging Face and then set it up by running:
huggingface-cli login
````

## 🚀 Running the Code

To train and run validation with the hybrid GNN–LLM pipeline, execute:

```bash
python3.11 main.py
```

This will load the configured dataset, fine-tune the LLM using LoRA, and produce predictions for molecular descriptors based on graph inputs.

## 🧪 User Interface (Optional)

An interface for testing the model is provided, created using Gradio. To use it, provide the model path weights in `configs/config.yaml` and run:

```bash
python3.11 UI/interface.py
```
Then visit http://127.0.0.1:7880 to access the interface.

**Note:**

* The interface currently supports **task selection and graph upload** via `.gxf` files.
* **Free-form Q\&A is not yet supported in the UI**, but the underlying fine-tuned model can be used independently for open-ended questions about molecular graphs.

## 📦 Required Dependencies

All dependencies are listed in `requirements.txt`. Key packages include:

* `torch==2.5.1`
* `torch-geometric==2.6.1`
* `transformers==4.53.2`
* `peft==0.13.2`
* `rdkit==2024.3.2`
* `hydra-core==1.3.2`
* `omegaconf==2.3.0`
* `gradio==5.23.2`
* `accelerate==1.0.1`
* `wandb==0.19.9`
* `scipy==1.10.1`
* `scikit-learn==1.3.2`
* `regex==2024.11.6`
* `numpy==1.24.4`
* `tqdm==4.67.1`
* `lxml==5.3.2`
* `wheel==0.45.1`

You can install them all with:

```bash
pip install -r requirements.txt
```

## 📁 Datasets and Models

* **Dataset**: We use the **AIDS Antiviral Screen Dataset**, where each molecule is represented as a graph with one-hot encoded atom and bond types.
* **Descriptors**: Computed via [`RDKit`](https://www.rdkit.org/) using SMILES strings translated from the graph.
* **Model**: The LLM used is [`microsoft/phi-1_5`](https://huggingface.co/microsoft/phi-1_5), fine-tuned using LoRA. Graph embeddings are injected into the LLM’s prompt stream using special tokens.
