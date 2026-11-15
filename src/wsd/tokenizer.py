"""One prompt/encoding implementation shared by training and inference."""

import torch


def format_prompt(example, tokenizer, metadata):
    example.validate()
    text = example.text
    strings = [text] + [c.definition for c in example.candidates]
    if any("*" in s or tokenizer.mask_token in s for s in strings):
        raise ValueError("literal target markers or mask tokens are unsupported by compact prompts")
    target = example.target
    marked = text[: target.start] + "*" + text[target.start : target.end] + "*" + text[target.end :]
    letters = metadata["letters"]
    choices = [f"{letters[i]}. {c.definition}" for i, c in enumerate(example.candidates)]
    choices.append(f"{letters[metadata['nota_index']]}. {metadata['none_text']}")
    return marked + "\n" + "\n".join(choices) + f"\n{metadata['separator']} {tokenizer.mask_token}"


def encode_examples(examples, tokenizer, metadata):
    prompts = [format_prompt(e, tokenizer, metadata) for e in examples]
    batch = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    if batch["input_ids"].shape[1] > metadata["max_length"]:
        raise ValueError("prompt exceeds max_length; shorten context without dropping target or candidates")
    masks = batch["input_ids"] == tokenizer.mask_token_id
    if not torch.all(masks.sum(1) == 1):
        raise ValueError("every prompt must contain exactly one answer mask")
    return dict(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], prediction_positions=masks.int().argmax(1)
    )
