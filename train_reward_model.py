import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import Dataset

# Load the ranked answers
df = pd.read_csv("q2_reward/answers.csv")

# Prepare the dataset for reward model training
def prepare_reward_dataset(df):
    # Create pairs of answers for comparison
    pairs = []
    prompts = df['prompt'].unique()
    
    for prompt in prompts:
        prompt_df = df[df['prompt'] == prompt]
        answers = prompt_df['answer'].tolist()
        ranks = prompt_df['rank'].tolist()
        
        # Create pairs where better ranked answers are chosen over worse ranked ones
        for i in range(len(answers)):
            for j in range(i+1, len(answers)):
                if ranks[i] < ranks[j]:  # Lower rank is better
                    pairs.append({
                        "prompt": prompt,
                        "chosen": answers[i],
                        "rejected": answers[j]
                    })
                else:
                    pairs.append({
                        "prompt": prompt,
                        "chosen": answers[j],
                        "rejected": answers[i]
                    })
    
    return Dataset.from_pandas(pd.DataFrame(pairs))

# Prepare the dataset
dataset = prepare_reward_dataset(df)

# Split into train and eval
dataset = dataset.train_test_split(test_size=0.2)

# Initialize model and tokenizer
model_name = "gpt2"  # You can change this to any base model you prefer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Configure the reward trainer
training_args = RewardConfig(
    output_dir="q2_reward/reward_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    max_length=512,
    max_steps=100,  # Train for 100 steps as required
    learning_rate=1e-5,
    report_to="none"
)

# Initialize the reward trainer
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("q2_reward/reward_model")
print("Reward model trained and saved to q2_reward/reward_model")