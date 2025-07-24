import time

import torch
import transformers
from loguru import logger


def hf_load_pretrained_llm(name, model_kwargs=None, tokenizer_kwargs=None,
                           model_class=transformers.AutoModelForCausalLM, tokenizer_class=transformers.AutoTokenizer,
                           cache_dir='.resources', device_map="auto"):
    if model_kwargs is None:
        model_kwargs = {}
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    logger.debug(f"Loading {name}-model...")
    start = time.time()
    model = model_class.from_pretrained(name, **model_kwargs, cache_dir=cache_dir, device_map=device_map)
    logger.success(
        f"Model {name} loaded {f'with args: {model_kwargs}' if model_kwargs is not None else ''}. ({time.time() - start:.2f}s)")
    logger.debug(f"Device map: {model.hf_device_map}")

    logger.debug(f"Loading {name}-tokenizer...")
    start = time.time()
    tokenizer = tokenizer_class.from_pretrained(name, **tokenizer_kwargs, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.success(
        f"Tokenizer {name} loaded {f'with args: {tokenizer_kwargs}' if tokenizer_kwargs is not None else ''}. ({time.time() - start:.2f}s)")

    return model, tokenizer


def model_to_device(model, args):
    if args.device == 'cuda:0':
        logger.debug(f"Moving {model.config._name_or_path} model to GPU...")
        start = time.time()
        model.to(args.device)

        logger.success(
            f"Successfully moved model '{model.config._name_or_path}' to GPU! ({time.time() - start:.2f}s) | "
            f"Available VRAM left: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024 ** 3):.2f} GB")