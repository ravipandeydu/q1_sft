import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os

# Set up the base model
model_name = "gpt2"  # You can change this to any base model you prefer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create directory structure
os.makedirs("q2_reward", exist_ok=True)
os.makedirs("q2_reward/reward_model", exist_ok=True)

# Define 5 prompts
prompts = [
    "Write a joke about artificial intelligence.",
    "Summarize the concept of machine learning in three sentences.",
    "Write a mini-essay about the future of technology.",
    "Explain quantum computing to a 10-year-old.",
    "Write a short poem about data science."
]

# Function to generate answers
def generate_answers(prompt, num_return_sequences=4):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate multiple sequences
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated sequences
    answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return answers

# Generate and rank answers
data = []

for prompt in prompts:
    print(f"Generating answers for: {prompt}")
    answers = generate_answers(prompt)
    
    # Display answers for manual ranking
    print("\nPlease rank the following answers from 1 (best) to 4 (worst):")
    for i, answer in enumerate(answers):
        print(f"\nAnswer {i+1}:\n{answer}")
    
    # Get rankings from user
    rankings = []
    while len(rankings) < 4:
        try:
            rank_input = input(f"\nEnter rank for Answer {len(rankings)+1} (1-4): ")
            rank = int(rank_input)
            if 1 <= rank <= 4 and rank not in rankings:
                rankings.append(rank)
            else:
                print("Please enter a unique number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Add to dataset
    for i, answer in enumerate(answers):
        data.append({
            "prompt": prompt,
            "answer": answer,
            "rank": rankings[i]
        })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("q2_reward/answers.csv", index=False)
print("\nAnswers saved to q2_reward/answers.csv")