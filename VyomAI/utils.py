# for model making utils
import torch.nn as nn
from typing import Dict, Tuple, List
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
                _ = model.generate(
                    input_ids, attention_mask=masks, generation_config=generation_config
                )
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
