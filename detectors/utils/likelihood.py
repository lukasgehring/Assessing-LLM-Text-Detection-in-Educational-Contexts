import torch


# Modified code from: DetectGPT
def get_lls(texts, base_model, base_tokenizer, args):
    lls = []
    for text in texts:
        if text == "":
            continue
        lls.append(get_ll(text, base_model, base_tokenizer, args))
    return lls

# Modified code from: DetectGPT
# Get the log likelihood of each text under the base_model
def get_ll(text, base_model, base_tokenizer, args):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(args.device)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()