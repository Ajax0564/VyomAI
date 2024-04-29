import torch.nn as nn
from typing import Dict, Tuple, List, Optional
import torch


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
            with torch.no_grad():
                logits = model(input_ids=idx).logits
        else:
            with torch.no_grad():
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


def generate_seq2seq(
    model: nn.Module,
    encoder_output: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    decoder_start: torch.Tensor,
    max_new_tokens: Optional[int] = 5,
    temperature: Optional[float] = 1.0,
    do_sample: Optional[bool] = False,
    top_k: Optional[int] = 10,
    use_cache: Optional[bool] = False,
) -> torch.Tensor:
    """

    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete

    the sequence max_new_tokens times, feeding the predictions back into the model each time.

    Most likely you'll want to make sure to be in model.eval() mode of operation for this.

    """
    idx = decoder_start

    idx_next = idx

    index = 0

    #     for cur_pos in range(min_promp, total_len)

    for _ in range(max_new_tokens):

        if use_cache:
            with torch.no_grad():
                logits = model(
                    encoder_output=encoder_output,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=idx_next,
                    use_cache=use_cache,
                    start_pos=index,
                ).logits
            logits = logits[:, -1] / temperature
        else:
            with torch.no_grad():
                logits = model(
                    encoder_output=encoder_output,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=idx,
                    use_cache=use_cache,
                ).logits
            if index == 0:  # start token
                logits = logits[:, -1] / temperature
            else:
                logits = logits[:, -1, :] / temperature

        probs = torch.nn.functional.softmax(logits, dim=-1)

        # either sample from the distribution or take the most likely element

        if do_sample:

            idx_next = torch.multinomial(probs, num_samples=1)

        else:

            _, idx_next = torch.topk(probs, k=1, dim=-1)

        # append sampled index to the running sequence and continue

        idx = torch.cat((idx, idx_next), dim=1)

        index = idx.size()[1] - 1  # model already have idx-1 kv-cache stored

    return idx
