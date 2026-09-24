# Overcoming Representation Collapse: Zero-Shot LLMs vs. Supervised Encoders in Propaganda Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cf0waFVSJywR9IySwPxIz9E8kWjRSxDi?usp=sharing)

**Zero-Click Reproducibility:** This pipeline is fully automated. You can click the "Open in Colab" badge above to launch the interactive notebook. Navigate to **Runtime > Change runtime type**, select the **T4 GPU** hardware accelerator, and click **Run All**. The notebook will automatically fetch the dataset from this repository, configure the fixed random seeds (`random_state=42`), and generate the final evaluation visualizations. No manual data uploads or Hugging Face authentication tokens are required.

## Overview
This repository contains the empirical code, datasets, and visualization scripts for comparing the performance of 4-bit quantized open-weight Large Language Models (LLMs) against traditional supervised encoders for fine-grained political propaganda detection. 

This project was developed for the NLP Master's Seminar evaluation at the University of Trier to demonstrate the failure modes of supervised encoders on highly skewed, low-resource datasets.

## Models & Dataset
- **Dataset:** 6,129 filtered spans from SemEval-2020 Task 11 (PTC corpus). The pipeline automatically generates a fixed 600-sample train / 75-sample test split.
- **Baseline Model:** `microsoft/deberta-v3-base` (Fine-tuned for 3 epochs to trigger representation collapse).
- **Zero-Shot LLM:** `Qwen/Qwen2.5-7B-Instruct` (Deployed via 4-bit NF4 quantization to bypass memory and API limits).

## Hardware Requirements
To run this pipeline successfully, the following hardware is required due to the memory overhead of the 7B parameter LLM:
- **GPU:** NVIDIA T4 GPU (or better) with at least 16GB VRAM.
- **RAM:** 12GB+ System RAM.

*Note: The entire pipeline fits comfortably within the constraints of the Google Colab Free Tier.*

## How to Replicate (Local Environment)
If you prefer to run the experiments locally rather than in Colab, ensure you have Python 3.10+ and a local NVIDIA GPU configured with appropriate CUDA drivers.
