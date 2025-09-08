import gymnasium as gym
from gymnasium import spaces
import re
import numpy as np
from bert_score import score

class GSM8KEnv(gym.Env):
    """
    A Gymnasium environment interface for the GSM8K dataset.
    The agent interacts by generating answers to math problems.
    """
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = []
        for item in dataset:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are given a math problem. You must first think step by step and then give the final answer."},
                    {"role": "user", "content": item['question']}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            gold = item['answer']
            reasoning, answer = parse_golden(gold)
            if reasoning is not None and answer is not None:
                formatted_gold = f"<think>{reasoning}</think><answer>{answer}</answer>"
            else:
                formatted_gold = gold  # Fallback if parsing fails
            self.dataset.append({'prompt': prompt, 'gold': formatted_gold})
        
        self.current_idx = 0
        self.max_length = 1000  

        self.observation_space = spaces.Text(max_length=self.max_length)

        self.action_space = spaces.Text(max_length=self.max_length)
        
    def reset(self, seed=None, options=None):
        """Reset the environment to a new problem."""
        super().reset(seed=seed)
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        obs = self.dataset[self.current_idx]['prompt']
        return obs, {}  

    def step(self, action):
        """Take a step with the generated answer."""
        gold_answer = self.dataset[self.current_idx]['gold']

        reward = compute_rewards([action], [gold_answer])[0]

        terminated = True
        truncated = False

        obs = None

        return obs, reward, terminated, truncated, {'gold': gold_answer, 'pred': action}

    def sample_batch(self, batch_size):
        """Sample a batch of problems for training."""
        indices = np.random.choice(len(self.dataset), batch_size, replace=False)
        indices = [int(i) for i in indices]
        prompts = [self.dataset[i]['prompt'] for i in indices]
        solutions = [self.dataset[i]['gold'] for i in indices]

        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt": prompts,
            "solution": solutions
        }

        return batch

def parse_answer(text, reasoning_start="<think>", reasoning_end="</think>", 
                    solution_start="<answer>", solution_end="</answer>"):
    """
    Extract reasoning and numeric answer from model-generated text.
    Returns: (reasoning:str or None, answer:int or None)
    """
    reasoning, answer = None, None
        
    if text is None:
        return reasoning, answer
        
    # Extract reasoning
    reasoning_match = re.search(
        re.escape(reasoning_start) + r"(.*?)" + re.escape(reasoning_end),
        text, re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        
    # Extract answer
    solution_match = re.search(
        re.escape(solution_start) + r"(.*?)" + re.escape(solution_end),
        text, re.DOTALL
    )
    if solution_match:
        # Extract numeric value from solution
        num_match = re.search(r"-?\d+", solution_match.group(1))
        if num_match:
            answer = int(num_match.group(0))
        
    return reasoning, answer

def parse_golden(text):
    """
    Extract reasoning and numeric answer from golden reference (GSM8K style)
    Returns: (reasoning:str or None, answer:int or None)
    """
    reasoning, answer = None, None
    if text is None:
        return reasoning, answer
        
    # Reasoning: everything before the answer marker
    reasoning = text.split("####")[0].strip() if "####" in text else None
        
    # Answer: number after '####'
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        answer = int(match.group(1))
        
    return reasoning, answer

def compute_rewards(responses, solutions):
    rewards = []
    for response, solution in zip(responses, solutions):

        pred_reasoning, pred_answer = parse_answer(response)
        gold_reasoning, gold_answer = parse_answer(solution)

        # 1) Correctness of final answer
        correctness = 1.0 if pred_answer == gold_answer else 0.0

        # 2) Format reward
        format_reward = 1.0 if (pred_reasoning is not None and pred_answer is not None) else 0.0

        # 3) Reasoning similarity using BERTScore (F1)
        if pred_reasoning and gold_reasoning:
            P, R, F1 = score(
                [pred_reasoning], [gold_reasoning],
                lang="en", rescale_with_baseline=True,
                model_type="distilbert-base-uncased"
            )
            reasoning_score = F1.item()
        else:
            reasoning_score = 0.0
            
        # Weighted combination
        weighted_reward = (
            0.15 * format_reward +
            0.7 * correctness +
            0.15 * reasoning_score
        )

        rewards.append(weighted_reward)

    return np.array(rewards, dtype=np.float32)
