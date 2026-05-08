# Vision Language Model (VLM) Fine-Tuning with QLoRA 🚀

This repository contains a comprehensive Jupyter Notebook demonstrating how to fine-tune a Vision Language Model (VLM) to convert document images into structured Markdown using QLoRA.

## 📖 Objective
The main goal of this project is to adapt a pretrained multimodal model to perform highly accurate Document-to-Markdown generation. Specifically, the notebook covers:
- **Multimodal Learning**: Handling both image and text inputs.
- **Parameter-Efficient Fine-Tuning (PEFT)**: Utilizing QLoRA (4-bit quantization + LoRA adapters) to train large models on consumer-grade hardware (e.g., Kaggle T4 GPUs).
- **Document Understanding**: Teaching the model to preserve headings, lists, tables, and equations from raw document images.
- **Inference & Evaluation**: Generating Markdown from unseen document images and comparing it with the ground truth.

## 🧠 Model & Dataset
- **Base Model**: [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) (A powerful 2-billion parameter vision-language model).
- **Dataset**: A subset of the Nougat Training Dataset (Document Images + Ground Truth Markdown).

## 🛠️ Technology Stack
- **PyTorch** & **Hugging Face Transformers**
- **PEFT (Parameter-Efficient Fine-Tuning)**
- **BitsAndBytes** (for 4-bit NF4 quantization)
- **qwen-vl-utils** (for processing vision inputs specifically for Qwen2-VL)

## 📋 Steps Covered in the Notebook
1. **Environment Setup**: Installing required libraries and authenticating with Hugging Face.
2. **Dataset Exploration**: Loading image-markdown pairs and visualizing ground truth data.
3. **ChatML Formatting**: Preparing the dataset into the `user`/`assistant` conversational format expected by instruct models.
4. **QLoRA Configuration**: Loading the model in 4-bit precision and applying LoRA adapters to linear layers (`q_proj`, `v_proj`, etc.) to significantly reduce memory footprint.
5. **Training**: Using Hugging Face `Trainer` with gradient accumulation and gradient checkpointing to train the adapters.
6. **Generation & Testing**: Passing unseen document images to the fine-tuned model and generating Markdown text.
7. **Model Saving & Pushing**: Saving the trained LoRA weights and pushing them to the Hugging Face Hub.

## 🚀 How to Run
1. Open the `Vision_Language_Model_Fine_Tuning.ipynb` notebook.
2. We recommend running this on a platform like **Kaggle** or **Google Colab** with access to GPU acceleration (e.g., T4, A100).
3. Follow the cells sequentially. Make sure to update the Hugging Face login tokens if you plan to push your own trained model to the Hub!

---
*This repository was created to showcase efficient VLM fine-tuning techniques for document parsing tasks.*
