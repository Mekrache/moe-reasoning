
import random
import numpy as np
from utils.dataset import compute_rewards
import torch

def left_pad(sequences, pad_value=0):
    """
    Left-pad a list of sequences to the same length.
    Args:
        sequences: List of tensors of shape (seq_len,)
        pad_value: Value to pad with (use tokenizer.pad_token_id for input_ids, 0 for masks/log_probs)
    Returns:
        Padded tensor of shape (batch_size, max_len)
    """
    if not sequences:
        return torch.empty(0, 0, dtype=torch.long)
    
    max_len = max(seq.shape[0] for seq in sequences)
    padded = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            pad_size = max_len - seq.shape[0]
            pad_tensor = torch.full((pad_size,), pad_value, dtype=seq.dtype, device=seq.device)
            padded_seq = torch.cat([pad_tensor, seq], dim=0)
        else:
            padded_seq = seq
        padded.append(padded_seq)
    return torch.stack(padded, dim=0)

def compute_logits(llm, tokenizer, accelerator, full_responses):
    full_attention_mask = (full_responses != tokenizer.pad_token_id).long()

    with accelerator.autocast(): 
        logits = llm(input_ids=full_responses, attention_mask=full_attention_mask, use_cache=False).logits
    
    logits = torch.clamp(logits, min=-1e4, max=1e4)

    log_probs = torch.log_softmax(logits, dim=-1)

    selected_log_probs = torch.gather(
        input= log_probs,
        dim=2,
        index=full_responses.unsqueeze(-1)
    ).squeeze(-1)

    return selected_log_probs 

def collect_experiences(llm, tokenizer, accelerator, batch, batch_size, num_rollouts):
    input_ids = batch["input_ids"].to(accelerator.device)
    attention_mask = batch["attention_mask"].to(accelerator.device)
    input_size = input_ids.shape[1]
    solution = batch["solution"]

    # Generate responses with current model
    with torch.no_grad():
        full_responses = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.5,
            top_p=0.95,
            num_return_sequences=num_rollouts,
            )
        
        assistant_responses = full_responses[:, input_size:]

        #Calculate logits
        log_probs = compute_logits(llm, tokenizer, accelerator, full_responses)

        decoded_responses = tokenizer.batch_decode(
            assistant_responses,
            skip_special_tokens=True
            ) 
        
        rewards = compute_rewards(decoded_responses, np.repeat(solution, num_rollouts))

        rewards = np.reshape(rewards, [batch_size, num_rollouts])
        
        # Add numerical stability to advantage calculation
        reward_mean = np.mean(rewards, axis=1, keepdims=True)
        reward_std = np.std(rewards, axis=1, keepdims=True)
        advantages = (rewards - reward_mean) / (reward_std + 1e-8)
        #advantages = np.clip(advantages, -5.0, 5.0)

        advantages = advantages.reshape(-1, 1)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(accelerator.device)

        for i in range(full_responses.shape[0]):
            if advantages[i].abs() < 0.01:
                advantages[i] = 0.0

        padded_tokens = (full_responses != tokenizer.eos_token_id).int()
        response_start_idx = padded_tokens.argmax(axis=-1)
        response_end_idx = padded_tokens.shape[1] - torch.flip(
            padded_tokens, dims=[1]
        ).argmax(dim=1)

        response_mask = torch.zeros_like(padded_tokens)
        for i in range(len(response_mask)):
            response_mask[i, input_size: response_end_idx[i]] = 1

    experience = [{
        "input_sequence": full_responses[
            i, response_start_idx[i]: response_end_idx[i]
        ],
        "log_probs": log_probs[
            i, response_start_idx[i]: response_end_idx[i]
        ],
        "response_mask": response_mask[
            i, response_start_idx[i]: response_end_idx[i]
        ],
        "advantages": advantages[i],
    } for i in range(advantages.shape[0]) if (advantages[i].abs() > 0.01)
    ]

    return experience, rewards.mean()


def calculate_grpo_loss(
        log_probs, 
        old_log_probs, 
        advantages, 
        response_mask, 
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2
        ):

    ratio = torch.exp(log_probs - old_log_probs)

    clipped_ratio = torch.clamp(
        ratio, 
        1.0 - clip_epsilon_low, 
        1.0 + clip_epsilon_high
    )

    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    loss = -torch.min(surrogate1, surrogate2)

    # Mask out padding / non-response tokens
    loss = loss * response_mask

    # Normalize by number of valid tokens
    loss = loss.sum() / (response_mask.sum() + 1e-8)

    return loss

def train_on_batch(llm, tokenizer, accelerator, experiences):
    # Simulate training on a batch
    full_sequence = left_pad(
        [b["input_sequence"] for b in experiences]).to(
            accelerator.device
        )
    
    attention_mask = left_pad(
        [torch.ones_like(b["input_sequence"]) for b in experiences], 0
    ).to(
        accelerator.device
    )

    old_log_probs = left_pad(
        [b["log_probs"] for b in experiences]).to(
            accelerator.device
        )
    response_mask = left_pad(
        [b["response_mask"] for b in experiences]).to(
            accelerator.device
        )

    advantages = (
        torch.cat([b["advantages"] for b in experiences], dim=0)
        .unsqueeze(-1)
        .to(accelerator.device)
    )

    log_probs = compute_logits(llm, tokenizer, accelerator, full_sequence)
    
    loss = calculate_grpo_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        #loss_implementation="bnpo"
    )
    return loss
