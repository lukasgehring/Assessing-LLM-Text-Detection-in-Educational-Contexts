from loguru import logger

def strip_newlines(text):
    return ' '.join(text.split())

def preprocess_data(data, mask_tokenizer, args):
    original = data["human"]
    generated = data["llm"]

    # remove whitespaces
    logger.debug("Strip whitespaces and newlines from human-written and llm-generated texts...")
    original = original.str.strip()
    generated = generated.str.strip()

    # remove newlines
    original = original.apply(strip_newlines)
    generated = generated.apply(strip_newlines)

    original = original.to_list()
    generated = generated.to_list()

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    logger.debug("Remove samples with more than 512 tokens.")

    tokenized_original = mask_tokenizer(original)
    tokenized_generated = mask_tokenizer(generated)

    mask = [len(x) <= 512 and len(y) <= 512 for x, y in
            zip(tokenized_original["input_ids"], tokenized_generated["input_ids"])]
    original = [value for i, value in enumerate(original) if mask[i]]
    generated = [value for i, value in enumerate(generated) if mask[i]]

    logger.info(f"Prepared data with {len(original[:args.n_samples])} human-written and llm-generated texts, with a maximum of 512 tokens according to mask_tokenizer.")

    args.indices = args.indices[mask]
    args.indices = args.indices[:args.n_samples]

    logger.info(f"Updated indices: {args.indices.values.tolist()}")

    return {'original': original[:args.n_samples], 'sampled': generated[:args.n_samples]}