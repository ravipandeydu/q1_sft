import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# Load the trained reward model
reward_model = AutoModelForSequenceClassification.from_pretrained("q2_reward/reward_model")
reward_tokenizer = AutoTokenizer.from_pretrained("q2_reward/reward_model")

# Load the base model for generating new answers
base_model_name = "gpt2"  # Use the same base model as before
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Function to generate new answers
def generate_new_answers(prompt, num_return_sequences=4):
    inputs = base_tokenizer(prompt, return_tensors="pt")
    
    # Generate multiple sequences with different temperatures
    answers = []
    temperatures = [0.5, 0.7, 1.0, 1.5]  # Different temperatures for diversity
    
    for temp in temperatures:
        outputs = base_model.generate(
            inputs["input_ids"],
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            pad_token_id=base_tokenizer.eos_token_id
        )
        
        answer = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers.append(answer)
    
    return answers

# Function to score answers using the reward model
def score_answers(prompt, answers):
    scores = []
    
    for answer in answers:
        inputs = reward_tokenizer(prompt, answer, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = reward_model(**inputs)
            score = outputs.logits[0].item()
        
        scores.append(score)
    
    return scores

# Define new prompts for evaluation
new_prompts = [
    "Write a joke about programming.",
    "Explain how neural networks work in simple terms.",
    "Write a short story about robots and humans becoming friends."
]

# Generate and score new answers
results = []

for prompt in new_prompts:
    print(f"\nEvaluating prompt: {prompt}")
    
    # Generate new answers
    answers = generate_new_answers(prompt)
    
    # Score the answers
    scores = score_answers(prompt, answers)
    
    # Display answers and scores
    for i, (answer, score) in enumerate(zip(answers, scores)):
        print(f"\nAnswer {i+1} (Score: {score:.4f}):\n{answer}")
        results.append({
            "prompt": prompt,
            "answer": answer,
            "score": score,
            "temperature": [0.5, 0.7, 1.0, 1.5][i]  # Corresponding temperature
        })

# Save results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("q2_reward/evaluation_results.csv", index=False)

# Visualize the results
plt.figure(figsize=(12, 8))

for i, prompt in enumerate(new_prompts):
    prompt_results = results_df[results_df['prompt'] == prompt]
    
    plt.subplot(len(new_prompts), 1, i+1)
    plt.bar(
        range(len(prompt_results)), 
        prompt_results['score'],
        tick_label=[f"Temp={t}" for t in prompt_results['temperature']]
    )
    plt.title(f"Prompt: {prompt[:30]}...")
    plt.ylabel("Reward Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("q2_reward/reward_scores.png")
plt.close()

# Create a summary of the evaluation
with open("q2_reward/summary.md", "w") as f:
    f.write("# Reward Model Evaluation Summary\n\n")
    f.write("## Overview\n")
    f.write("This document summarizes the evaluation of the trained reward model on new prompts and answers.\n\n")
    
    f.write("## Evaluation Results\n")
    for prompt in new_prompts:
        f.write(f"### Prompt: {prompt}\n\n")
        
        prompt_results = results_df[results_df['prompt'] == prompt].sort_values('score', ascending=False)
        
        f.write("| Rank | Temperature | Score | Answer Preview |\n")
        f.write("|------|-------------|-------|---------------|\n")
        
        for i, (_, row) in enumerate(prompt_results.iterrows()):
            answer_preview = row['answer'][:50].replace("\n", " ") + "..."
            f.write(f"| {i+1} | {row['temperature']} | {row['score']:.4f} | {answer_preview} |\n")
        
        f.write("\n")
    
    f.write("## Conclusion\n")
    f.write("The evaluation shows how the reward model scores different answers generated with varying temperatures. ")
    f.write("Higher scores should correlate with better answers according to the preferences captured during training.\n")

print("\nEvaluation completed. Results saved to q2_reward/evaluation_results.csv")
print("Visualization saved to q2_reward/reward_scores.png")
print("Summary saved to q2_reward/summary.md")