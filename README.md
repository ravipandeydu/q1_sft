# Tiny Supervised Fine-Tuning (SFT) Project

This project demonstrates how to fine-tune a large language model (LLM) to create a polite AI assistant using Supervised Fine-Tuning (SFT). The implementation uses LoRA (Low-Rank Adaptation) with PEFT (Parameter-Efficient Fine-Tuning) to efficiently adapt a pre-trained model with minimal computational resources.

## Project Overview

The goal of this project is to transform a base language model into a polite, helpful assistant through supervised fine-tuning on a carefully curated dataset of prompt-response pairs. The dataset includes examples of factual Q&A, polite responses, length-controlled answers, and appropriate refusals for problematic requests.

## Files in this Repository

- `dataset.jsonl`: Contains the training dataset with prompt-response pairs
- `train.py`: Script for fine-tuning the model using LoRA with PEFT
- `before_after.md`: Comparison of model responses before and after fine-tuning
- `README.md`: This file, providing project documentation

## Dataset

The dataset consists of 30 prompt-response pairs covering various scenarios:

- Factual Q&A examples (e.g., "Capital of France?")
- Polite-tone examples (e.g., "Please translate...")
- Short-form vs. long-form answers (demonstrating length control)
- Refusal cases (e.g., illicit requests that receive safe denials)

Each pair is formatted with `<|user|>` and `<|assistant|>` tokens to clearly delineate the roles.

## Model & Training

The project uses the TinyLlama/TinyLlama-1.1B-Chat-v1.0 model by default (optimized for systems with limited VRAM) and fine-tunes it using LoRA with PEFT. The training configuration includes:

- 3-5 epochs
- Learning rate of approximately 5e-5
- 4-bit quantization for efficient training
- LoRA rank of 8
- Gradient checkpointing for memory efficiency

## Running the Code Locally

### Prerequisites

1. Python 3.8 or higher
2. PyTorch 2.0 or higher
3. Hugging Face account with access to models
4. GPU with at least 4GB VRAM

### Installation

```bash
# Create a virtual environment
python -m venv sft_env

# Activate the environment
# On Windows
sft_env\Scripts\activate
# On macOS/Linux
source sft_env/bin/activate

# Install dependencies
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

### Fine-tuning the Model

```bash
python train.py --dataset_path "dataset.jsonl" --output_dir "./output" --epochs 3 --learning_rate 5e-5
```

The default configuration uses TinyLlama/TinyLlama-1.1B-Chat-v1.0 which works well on GPUs with 4GB VRAM. For systems with more GPU memory, you can use a larger model:

```bash
python train.py --model_name "NousResearch/Meta-Llama-3-8B" --dataset_path "dataset.jsonl" --output_dir "./output" --epochs 3 --learning_rate 5e-5 --batch_size 4 --gradient_accumulation_steps 4
```

### Memory Monitoring

For systems with limited VRAM, you can monitor GPU memory usage during training:

```bash
python train.py --monitor_gpu
```

This will display memory usage at key points during the training process, helping you identify potential bottlenecks and optimize accordingly.

### Evaluating the Model

After training, you can evaluate the model by comparing its responses to the same prompts before and after fine-tuning. The `before_after.md` file provides examples of such comparisons.

## Customization

You can customize this project by:

1. Adding more examples to the dataset.jsonl file
2. Adjusting the training parameters in train.py
3. Using a different base model
4. Modifying the LoRA configuration for different fine-tuning characteristics

## Memory Optimization Techniques

This project implements several memory optimization techniques to enable training on GPUs with limited VRAM (4GB):

1. **4-bit Quantization**: Reduces model precision to save memory
2. **LoRA Fine-tuning**: Updates only a small subset of model parameters
3. **Gradient Checkpointing**: Trades computation for memory by recomputing activations during backpropagation
4. **Small Batch Size**: Default batch size of 2 with gradient accumulation of 8 steps
5. **Memory Efficient Optimizers**: Uses optimized PyTorch implementations
6. **Garbage Collection**: Actively frees unused memory during training
7. **GPU Memory Monitoring**: Optional feature to track memory usage

## Notes

- Training time will vary based on your hardware. On a consumer GPU with 4GB VRAM, expect 1-2 hours for the TinyLlama model.
- The model's performance will depend on the quality and diversity of the training examples.
- For production use, more extensive training data and evaluation would be recommended.
- If you encounter out-of-memory errors, try reducing the batch size further or increasing gradient accumulation steps.