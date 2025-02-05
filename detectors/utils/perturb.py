import re

import numpy as np
from tqdm import tqdm

# Modified code from: DetectGPT
def tokenize_and_mask(text, span_length, pct, args):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    n_spans = int(n_spans)

    # Added to prevent infinite regeneration loop
    if n_spans > 28:
        n_spans = 28

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

# Modified code from: DetectGPT
def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# Modified code from: DetectGPT
# replace each masked span with a sample from T5 mask_model
def replace_masks(mask_tokenizer, mask_model, texts, args):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)

    # set stop token if model execute with </s>
    ids = (outputs == stop_id).any(dim=1).flatten()
    if not ids.all():
        idx = (outputs[(~ids).nonzero()] == 1).nonzero()[0]
        outputs[(~ids).nonzero(),idx] = stop_id
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

# Modified code from: DetectGPT
def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    # define regex to match all <extra_id_*> tokens, where * is an integer
    pattern = re.compile(r"<extra_id_\d+>")
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

# Modified code from: DetectGPT
def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

# Modified code from: DetectGPT
def perturb_texts_(texts, span_length, pct, mask_tokenizer, mask_model, args):
    masked_texts = [tokenize_and_mask(x, span_length, pct, args) for x in texts]
    raw_fills = replace_masks(mask_tokenizer, mask_model, masked_texts, args)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, args) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(mask_tokenizer, mask_model, masked_texts, args)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

        # Add reduce inference time (following DetectLLM)
        if attempts > args.max_num_attempts:
            break

    return perturbed_texts

# Modified code from: DetectGPT
def perturb_texts(texts, span_length, pct, mask_model, mask_tokenizer, args):
    chunk_size = args.chunk_size

    outputs = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, mask_tokenizer, mask_model, args))
    return outputs