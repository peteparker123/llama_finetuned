# Medical Bot - Llama 2 Fine-Tuned AI Assistant

A specialized medical AI assistant built by fine-tuning Llama-2-7b using QLoRA and SFT (Supervised Fine-Tuning) for medical question answering.

## Overview

This project fine-tunes Meta's Llama-2-7b-chat model to create a specialized medical AI assistant that provides reliable health and medical information. The model is trained on curated medical Q&A data using efficient quantization and LoRA adapters.

## Features

**Specialized Medical Knowledge** - Fine-tuned specifically for medical questions  
**Memory Efficient** - Uses 4-bit quantization and QLoRA  
**Quick Training** - Efficient training with LoRA adapters  
**Scope Control** - Trained to refuse non-medical questions  
**Interactive** - Can be used in interactive conversation mode  
**Pre-trained Model Available** - Download from Hugging Face Hub

## Technical Stack

- **Base Model**: Llama-2-7b-chat-hf (NousResearch)
- **Fine-tuning Method**: QLoRA (Quantized LoRA)
- **Quantization**: 4-bit (NF4)
- **Training Framework**: TRL (Transformers Reinforcement Learning)
- **Optimization**: LoRA + 4-bit quantization

## Requirements

```bash
pip install datasets
pip install accelerate==0.21.0
pip install peft==0.4.0
pip install bitsandbytes==0.40.2
pip install transformers==4.31.0
pip install trl==0.4.7
pip install torch
```

## Dataset

The model is trained on custom medical Q&A data covering:
- Common symptoms and diagnoses
- Treatment recommendations
- Disease information (malaria, arthritis, etc.)
- Health and fitness advice
- Scope limitation (refuses non-medical questions like sports, movies, etc.)

Example training data includes:
- Throat infections and fever
- Respiratory conditions
- Back pain and spinal issues
- Joint inflammation
- General exercise benefits

## Model Configuration

### Quantization Settings
```python
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
```

### LoRA Parameters
```python
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
```

### Training Settings
- **Epochs**: 10
- **Learning Rate**: 2e-4
- **Batch Size**: 4 (per device)
- **Gradient Accumulation Steps**: 1
- **Output Directory**: ./results

## Usage

### 1. Training the Model
```python
# Install dependencies
# Run all cells in the notebook

# The model will be trained and saved as 'medical_bot-llama2'
```

### 2. Interactive Testing
```python
# Use the interactive mode to chat with the medical bot
system = "You are a specialized medical AI assistant. Only answer medical questions."
# Follow the interactive prompt to ask medical questions
# Type 'quit' to exit
```

### 3. Using Pre-trained Model
```python
# Download from Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "peteparker456/medical_diagnosis_llama2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
```

## Example Interaction

**Input:**
```
System: You are a specialized medical AI assistant. Only answer medical questions.
Human: I am 45 years old and have a sore throat, swollen lymph nodes, and a fever. What might be wrong with me?
```

**Output:**
```
Assistant: These symptoms are consistent with acute pharyngitis or tonsillitis, often caused by viral or bacterial infection. You should see a healthcare provider for proper diagnosis and treatment...
```

## Model Features

### Scope Control
The model is trained to:
- ✅ Answer medical questions in detail
- ✅ Provide health and fitness advice
- ❌ Refuse non-medical questions (sports, movies, entertainment)
- ❌ Provide only general information (not a replacement for professional medical advice)

## Deployment

### Download Pre-trained Model
The trained model is available at:
👉 [peteparker456/medical_diagnosis_llama2](https://huggingface.co/peteparker456/medical_diagnosis_llama2)

### Save Locally
```python
trainer.model.save_pretrained("medical_bot-llama2")
```

## Important Disclaimer

⚠️ **This AI assistant is for informational purposes only.**

- Not a substitute for professional medical advice
- Always consult qualified healthcare professionals for diagnosis and treatment
- Use for educational and informational purposes only
- Do not rely solely on this system for medical decisions

## Performance Metrics

- **Training Framework**: TRL with Hugging Face Transformers
- **Quantization Method**: QLoRA (4-bit NF4)
- **Memory Efficiency**: ~70% reduction in memory usage
- **Training Time**: Efficient with gradient accumulation

## Monitoring & Evaluation

The notebook includes TensorBoard integration for monitoring training:
```python
%load_ext tensorboard
%tensorboard --logdir results/runs
```

## Project Structure

```
├── notebook.ipynb          # Main training notebook
├── results/                # Training outputs and checkpoints
├── medical_bot-llama2/     # Saved model
└── README.md               # This file
```

## What This Project Teaches

- Fine-tuning large language models with QLoRA
- 4-bit quantization for memory efficiency
- Supervised Fine-Tuning (SFT) with TRL
- Prompt engineering and system instructions
- Building specialized AI assistants
- Hugging Face model deployment

## Author

Created by a college student as a project to demonstrate fine-tuning large language models for specialized tasks note this project was developed in 2024

---

**For questions, improvements, or to download the pre-trained model, visit the [Hugging Face Hub](https://huggingface.co/peteparker456/medical_diagnosis_llama2)**

**Remember**: This is an educational project. Always consult healthcare professionals for medical advice! 🏥
