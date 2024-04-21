# for model making utils
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
import torch
from tqdm import tqdm


def model_size(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def init_weights(module: nn.Module) -> None:
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # copied from Transformer
        module.weight.data.normal_(mean=0.0, std=0.02)  # initializer_range": 0.02
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def model_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return {"total_params": total_params, "trainable_params": trainable_params}


def timing_cuda(
    model: nn.Module,
    num_batches: int,
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    is_decoder: bool = False,
    generation_config=None,
    device: str = "cuda",
) -> Tuple[float, float]:
    if is_decoder:
        model.generation_config.eos_token_id = None

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # We need NOT call torch.cuda.empty_cache() here as it appears to negate the warmup.

    latencies = []
    for _ in tqdm(range(num_batches)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        if is_decoder:
            with torch.no_grad():
                _ = model(input_ids, attention_mask=masks)
        else:
            with torch.no_grad():
                _ = model(input_ids, masks)
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        # print(f"\nLatency per token: {latency_ms / generation_config.min_new_tokens:.3f} ms")
        latencies.append(latency_ms)
        if is_decoder:
            print(
                f"\nLatency per token: {latency_ms / generation_config.min_new_tokens:.3f} ms"
            )
            # min_new_tokens (int, optional) — The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    max_memory = torch.cuda.max_memory_allocated(device=device) * 1e-6  # in MB

    return torch.mean.mean(latencies), max_memory  # time =  milisecond


def generate(
    model: nn.Module,
    tokenize_text: torch.Tensor,
    max_new_tokens: Optional[int] = 3,
    temperature: Optional[float] = 1.0,
    do_sample: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> torch.Tensor:
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (1,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    #     text = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    idx = tokenize_text
    idx_next = idx
    index = 0
    take = -1
    #     for cur_pos in range(min_promp, total_len)
    for _ in range(max_new_tokens):
        if use_cache == False:
            logits = model(input_ids=idx).logits
        else:
            logits = model(
                input_ids=idx_next, start_pos=index, use_cache=use_cache
            ).logits

        if take != 0:
            logits = logits[:, take, :] / temperature
            if use_cache == True:
                take = 0
        else:
            logits = logits[:, -1] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        idx = torch.cat((idx, idx_next), dim=1)
        index = idx.size()[1] - 1  # model already have idx-1 kv-cache stored

    return idx
