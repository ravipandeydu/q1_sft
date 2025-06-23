# Reward Model Training Project

## Overview

This project implements a reward model training pipeline based on human preferences. The goal is to capture preferences in code and train a model that can score answers according to those preferences.

## Project Structure

```
q2_reward/
├── answers.csv                # CSV file with prompts, answers, and ranks
├── reward_model/              # Saved HuggingFace reward model
├── evaluation_results.csv     # Results of evaluating the reward model
├── reward_scores.png          # Visualization of reward scores
└── summary.md                 # Evaluation summary
```

## Components

### 1. Generate Answers (`generate_answers.py`)

This script:
- Defines 5 prompts (jokes, summaries, mini-essays, etc.)
- Generates 4 candidate answers per prompt using a base language model
- Allows manual ranking of answers from 1 (best) to 4 (worst)
- Saves the results to `q2_reward/answers.csv` in the format: prompt, answer, rank

### 2. Train Reward Model (`train_reward_model.py`)

This script:
- Loads the ranked answers from `q2_reward/answers.csv`
- Prepares a dataset for reward model training by creating comparison pairs
- Uses HuggingFace's `trl` library and `RewardTrainer` to train the model
- Trains for 100 steps as specified in the requirements
- Saves the trained model to `q2_reward/reward_model/`

### 3. Evaluate Reward Model (`evaluate_reward_model.py`)

This script:
- Loads the trained reward model
- Generates new answers for a set of test prompts
- Scores the answers using the reward model
- Visualizes the reward scores
- Creates a summary of the evaluation

### 4. Analysis Notebook (`analyse.ipynb`)

This Jupyter notebook:
- Demonstrates the entire workflow from data preparation to evaluation
- Provides visualizations and analysis of the reward model's performance
- Checks if higher scores correlate with better answers

## How to Run

1. Generate and rank answers:
   ```
   python generate_answers.py
   ```

2. Train the reward model:
   ```
   python train_reward_model.py
   ```

3. Evaluate the reward model:
   ```
   python evaluate_reward_model.py
   ```

4. Open the analysis notebook for detailed exploration:
   ```
   jupyter notebook analyse.ipynb
   ```

## Results

After running the evaluation script, you can check:

1. The `q2_reward/evaluation_results.csv` file for detailed scores
2. The `q2_reward/reward_scores.png` visualization to see how different answers are scored
3. The `q2_reward/summary.md` file for a summary of the evaluation

The analysis should verify that higher reward scores correlate with better answers according to the preferences captured during training.