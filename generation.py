import torch
from typing import List, Optional

def _unwrap_for_generate(model):
    if hasattr(model, "module"):  # DDP
        return model.module
    return model

@torch.no_grad()
def generate_n(
    model,
    tokenizer,
    prompts: List[str],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    repetition_penalty: float = 1.0,
) -> List[List[str]]:
    """
    Batched generation for decoder-only LMs.
    Returns grouped outputs per prompt.
    """
    gen_model = _unwrap_for_generate(model)
    
    # Temporarily disable gradient checkpointing for generation
    was_training = gen_model.training
    grad_ckpt_enabled = getattr(gen_model.config, 'gradient_checkpointing', False)
    if grad_ckpt_enabled and hasattr(gen_model, 'gradient_checkpointing_disable'):
        gen_model.gradient_checkpointing_disable()
    
    gen_model.eval()

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True).to(device)

    # If temperature is 0 or None, do greedy decoding.
    do_sample = temperature is not None and temperature > 1e-5
    
    # Ensure we have valid eos_token_id
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.convert_tokens_to_ids('</s>') if '</s>' in tokenizer.get_vocab() else None

    # For greedy decoding (do_sample=False), we can't use num_return_sequences > 1
    # So we generate n times in a loop instead
    if not do_sample:
        all_texts = []
        for _ in range(n):
            out = gen_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=1.0,  # Disable for greedy
                use_cache=True,
                no_repeat_ngram_size=0,
            )
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            all_texts.append(texts)
        
        # Transpose: [iteration][prompt_idx] -> [prompt_idx][iteration]
        grouped = []
        for i in range(len(prompts)):
            grouped.append([all_texts[j][i] for j in range(n)])
    else:
        # For sampling, use num_return_sequences but chunk into batches of at most
        # _SAMPLE_CHUNK to avoid OOM when n is large (e.g. k_max=32 with batched prompts).
        _SAMPLE_CHUNK = 8
        grouped = [[] for _ in range(len(prompts))]
        remaining = n
        while remaining > 0:
            chunk = min(remaining, _SAMPLE_CHUNK)
            out = gen_model.generate(
                **inputs,
                do_sample=True,
                temperature=max(temperature, 0.1),  # Avoid too low temp
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=chunk,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                use_cache=True,
            )
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            # HF groups as [p0_0..p0_{chunk-1}, p1_0..] → split per prompt
            for i in range(len(prompts)):
                grouped[i].extend(texts[i * chunk : (i + 1) * chunk])
            remaining -= chunk
    
    # Restore gradient checkpointing if it was enabled
    if grad_ckpt_enabled and hasattr(gen_model, 'gradient_checkpointing_enable'):
        gen_model.gradient_checkpointing_enable()
    
    # Restore training mode if it was on
    if was_training:
        gen_model.train()
    
    return grouped

def count_new_tokens(tokenizer, prompt: str, full_text: str) -> int:
    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
    f_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    return max(int(f_ids.shape[0] - p_ids.shape[0]), 0)
