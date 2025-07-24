import pandas as pd
from matplotlib import pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from accelerate.test_utils.testing import get_backend
import torch
from tqdm import tqdm
import seaborn as sns
from database.interface import get_answers

device, _, _ = get_backend()  # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model_id = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir="../.resources").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id, cache_dir="../.resources")

sns.set_theme(context="paper", style=None, font_scale=1, rc={
    # lines
    "lines.linewidth": 2,

    # grid
    "grid.linewidth": .5,

    # legend
    'legend.handletextpad': .5,
    'legend.handlelength': 1.0,
    'legend.labelspacing': 0.5,

    # axes
    'axes.spines.right': False,
    'axes.spines.top': False,

    # yticks
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.4,

    # save
    'savefig.format': 'pdf'
})

results = []
llm = "gpt-4o-mini-2024-07-18"
for is_human in [True, False]:
    for dataset in ["BAWE", "persuade", "argument-annotated-essays"]:
        df = get_answers(database="../../database/database.db", dataset=dataset, is_human=is_human,
                         generative_model=llm, prompt_mode="task")

        for i, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
            encodings = tokenizer(row.answer, return_tensors="pt")

            max_length = model.config.n_positions
            stride = 512
            seq_len = encodings.input_ids.size(1)

            nll_sum = 0.0
            n_tokens = 0
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss

                # Accumulate the total negative log-likelihood and the total number of tokens
                num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
                batch_size = target_ids.size(0)
                num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
            ppl = torch.exp(avg_nll)

            results.append({
                "dataset": dataset,
                "is_human": is_human,
                "score": ppl.item(),
                "sample": i
            })

df = pd.DataFrame(results)
fig, axs = plt.subplots(ncols=len(df.dataset.unique()), figsize=(7.5, 2.5))
for dataset, ax in zip(df.dataset.unique(), axs):
    for (_is_human, _dataset), sub_df in df[df.dataset != dataset].groupby(["is_human", "dataset"]):
        sns.kdeplot(sub_df, x="score", color="gray", fill=True, alpha=0.2, linewidth=.5, ax=ax)
    sns.kdeplot(df[df.dataset == dataset], x="score", hue="is_human", fill=True, alpha=0.8, linewidth=2, ax=ax)
    ax.set_title(dataset)
    ax.set_xlim(0, 50)
sns.despine(offset=0, trim=True)
plt.tight_layout()
plt.savefig(f"plots/gpt2_text_probability_{llm}.pdf")
plt.show()
