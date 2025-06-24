import os
import torch
import json
import argparse
import transformers
import gc
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model on a text dataset")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model to fine-tune")
    parser.add_argument("--dataset_path", type=str, default="dataset.jsonl", help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--monitor_gpu", action="store_true", help="Monitor GPU memory usage during training")
    return parser.parse_args()

def prepare_dataset(dataset_path):
    # Load the dataset from the JSONL file
    dataset = load_dataset('json', data_files=dataset_path)
    return dataset

def print_gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            free = reserved - allocated
            print(f"GPU {i}: Reserved: {reserved:.2f}MB, Allocated: {allocated:.2f}MB, Free: {free:.2f}MB")
    else:
        print("No GPU available")

def train():
    args = parse_args()
    
    # Set up quantization configuration for 4-bit training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Set torch options for memory efficiency
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    gc.collect()  # Run garbage collector to free memory
    
    if args.monitor_gpu:
        print("Initial GPU memory usage:")
        print_gpu_memory_usage()
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use fp16 precision
        low_cpu_mem_usage=True,     # Optimize CPU memory usage
    )
    
    if args.monitor_gpu:
        print("\nGPU memory usage after loading model:")
        print_gpu_memory_usage()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Set up LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    if args.monitor_gpu:
        print("\nGPU memory usage after applying PEFT:")
        print_gpu_memory_usage()
    
    # Print trainable parameters
    print("\nTrainable parameters:")
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"{name}: {param.shape}")
    print(f"\nTrainable params: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% of {all_params:,d} total params)")
    
    # Prepare dataset
    dataset = prepare_dataset(args.dataset_path)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        optim="adamw_torch",  # Use memory-efficient optimizer
    )
    
    # Set up SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        dataset_text_field="conversations",
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
    )
    
    if args.monitor_gpu:
        print("\nGPU memory usage before training:")
        print_gpu_memory_usage()
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    if args.monitor_gpu:
        print("\nGPU memory usage after training:")
        print_gpu_memory_usage()
    
    # Save the model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    train()