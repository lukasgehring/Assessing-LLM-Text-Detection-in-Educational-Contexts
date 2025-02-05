import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Modified code from: DetectLLM
def get_ranks(texts, base_model, base_tokenizer, args, log=True):
    return [get_rank(text, base_model, base_tokenizer, args, log=log) for text in texts]

# Modified code from: DetectGPT
# get the average rank of each observed token sorted by model likelihood
def get_rank(text, base_model, base_tokenizer, args, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(args.device)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

# Modified code from: DetectGPT
def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

# Modified code from: DetectGPT
def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)