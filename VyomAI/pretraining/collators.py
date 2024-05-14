import torch
from typing import Optional, Tuple
from torch.utils.data import Dataset
import pickle
import os
import time


def masked_language_modeling(
    input_ids: torch.Tensor,
    tokenizer,
    fraction: Optional[float] = 0.15,
    ignore_index: Optional[int] = -100,
) -> Tuple[torch.Tensor]:
    label = input_ids.clone()
    input_ids = input_ids.clone()

    special_tokens_mask = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in label.tolist()
        ],
        dtype=torch.bool,
    ).to(
        label.device
    )  # get all special token mask to ignore their selection in MLM
    probability_matrix = torch.full(label.shape, fraction).to(
        label.device
    )  # gen probability matrix to select 15% tokens for MLM
    probability_matrix.masked_fill_(
        special_tokens_mask, value=0.0
    )  # zero out the probability of special tokens so that they do not get selected
    # https://pytorch.org/docs/stable/generated/torch.bernoulli.html
    # Draws binary random numbers (0 or 1) from a Bernoulli distribution.
    masked_indices = torch.bernoulli(probability_matrix).bool().to(label.device)

    label[~masked_indices] = (
        ignore_index  # We only compute loss on masked tokens cross entropy ignore_index
    )

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(label.shape, 0.8)).bool().to(label.device)
        & masked_indices
    )  # rondomly select 80% tokens from 15% token
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token
    )  # replace it with [MASK]

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(label.shape, 0.5)).bool().to(label.device)
        & masked_indices
        & ~indices_replaced
    )  # from 15% ignore 80% select 10% from 20%
    random_words = torch.randint(len(tokenizer), label.shape, dtype=torch.long).to(
        label.device
    )  # get random token index from tokenizer
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged do nothing
    return input_ids, label, masked_indices


def log(t, eps=1e-9) -> torch.Tensor:
    "get log of input with added epsilon"
    return torch.log(t + eps)


def noise(t) -> torch.Tensor:
    "generate noise for given tensor"
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def sample(t, temperature=1.0) -> torch.Tensor:
    "for sampling the tokens with added noise and tempreture"
    return ((t / temperature) + noise(t)).argmax(dim=-1)


def electra(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    masked_indices: torch.Tensor,
    temperature: int = 3,
) -> Tuple[torch.Tensor]:
    sample_logits = logits[masked_indices]  # get token index from MLM step
    sampled = sample(
        sample_logits, temperature=temperature
    )  # temperature to control how many token to replace
    # scatter the sampled values back to the input
    discriminator_input = input_ids.clone()  # Original input
    discriminator_input[masked_indices] = sampled.detach()

    # generate discriminator labels, with replaced as True and original as False
    disc_labels = (input_ids != discriminator_input).float().detach()

    # get discriminator predictions of replaced / original
    non_padded_indices = torch.nonzero(
        input_ids != tokenizer.pad_token_id, as_tuple=True
    ).to(
        logits.device
    )  # needed to caluclate loss only on  non [PAD]  tokens
    return discriminator_input, disc_labels, non_padded_indices


class LanguageModeling(Dataset):

    def __init__(self, tokenizer, file_path: str, block_size: int):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        saved = False
        cache_dir = None
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        if os.path.exists(cached_features_file) and saved:
            start = time.time()
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            print(
                f"Loading features from cached file {cached_features_file} [took %.3f s]",
                time.time() - start,
            )

        else:
            print(f"Creating features from dataset file at {directory}")

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(
                0, len(tokenized_text) - block_size + 1, block_size
            ):  # Truncate in block of block_size
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[i : i + block_size]
                    )
                )
            # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            start = time.time()
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # saved = True
            print(
                f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return {"input_ids": torch.tensor(self.examples[i], dtype=torch.long)}
