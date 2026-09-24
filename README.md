# Overcoming Representation Collapse: Zero-Shot LLMs vs. Supervised Encoders in Propaganda Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wmRWHchTeqZCeVZbL4EC0UWuXC1X_zTC?usp=sharing)

**How to replicate this study:** You can click the "Open in Colab" badge above to launch the interactive notebook directly in your browser. Alternatively, you can copy the contents of the `.ipynb` file from this repository into a new Google Colab notebook. For the code to execute successfully, please ensure you navigate to **Runtime > Change runtime type** in Colab and select the **T4 GPU** hardware accelerator before running the cells.

## Overview
This repository contains the empirical code, datasets, and visualization scripts for comparing the performance of 4-bit quantized open-weight Large Language Models (LLMs) against traditional supervised encoders for fine-grained political propaganda detection. 

This project was developed in Google Colab, version-controlled via VS Code, and published here for strict reproducibility as part of the NLP Master's Seminar evaluation at the University of Trier.
This project was developed in Google Colab, version-controlled via VS Code, and published here for strict reproducibility as part of the NLP Master's Seminar evaluation at the University of Trier.

## Models & Dataset
- **Dataset:** SemEval-2020 Task 11 (PTC corpus) utilizing 120-character contextual framing windows.
- **Baseline Model:** `microsoft/deberta-v3-base` (Fine-tuned for 3 epochs on a 600-sample split).
- **Zero-Shot LLM:** `Qwen/Qwen2.5-7B-Instruct` (Deployed via 4-bit NF4 quantization to bypass API limits).

## Hardware Requirements
To run this pipeline successfully, the following hardware is required:
- **GPU:** NVIDIA T4 GPU (or better) with at least 16GB VRAM.
- **RAM:** 12GB+ System RAM.
- **Storage:** ~15GB free space to download model weights.

*Note: The entire pipeline runs within the constraints of the Google Colab Free Tier.*

## How to Replicate (Google Colab - Recommended)
1. Click the **Open in Colab** badge above.
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Go to **Runtime > Run all**. The notebook will install dependencies, run inference, evaluate both models, and generate the final visualization figures.

## How to Replicate (Local VS Code / Jupyter)
Ensure you have Python 3.10+ and a local NVIDIA GPU with appropriate CUDA drivers configured.

**1. Clone the repository:**
```bash
git clone [https://github.com/manjunath-leo/NLP-Propaganda-Detection.git](https://github.com/manjunath-leo/NLP-Propaganda-Detection.git)
cd NLP-Propaganda-Detection
2. Create and activate a virtual environment:
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install transformers accelerate bitsandbytes scikit-learn pandas matplotlib seaborn fpdf pypdf
