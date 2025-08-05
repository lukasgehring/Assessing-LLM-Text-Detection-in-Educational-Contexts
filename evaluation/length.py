import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.colors as mcolors
from database.interface import get_predictions
from evaluation.threshold import get_threshold
from evaluation.utils import select_best_roberta_checkpoint, remove_rows_by_condition, map_dipper_to_generative_model, get_roc_auc

sns.set_theme(context="paper", style=None, font_scale=1, rc={
    # lines
    "lines.linewidth": 2,

    # grid
    # "grid.linewidth": .5,

    # legend
    'legend.handletextpad': .5,
    'legend.handlelength': 1.0,
    'legend.labelspacing': 0.5,

    # axes
    'axes.spines.right': False,
    'axes.spines.top': False,

    # yticks
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.3,

    # save
    'savefig.format': 'pdf',

    # font
    "font.family": "Liberation Sans",
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})


def pastel_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([x + (1 - x) * amount for x in c])


custom_palette = ["#E69F00", "#0072B2", "#009E73", "#D55E00"]
pastel_colors = [pastel_color(c, 0.2) for c in custom_palette]
sns.set_palette(pastel_colors)

METRIC = "roc-auc"
scores = []

df = get_predictions(max_words=None, prompt_mode="task", )
df = select_best_roberta_checkpoint(df)
df = remove_rows_by_condition(df, conditions={
    "detector": ["gpt-zero", "intrinsic-dim"],
    "name": "mixed"
})

# make sure, to load the correct dipper results
df = df.apply(map_dipper_to_generative_model, axis=1)

max_words_list = [-1, 50, 100, 150, 200, 250]
for (detector,), sub_df in df.groupby(['detector']):
    # for prompt_mode in sub_df.prompt_mode.unique():
    for max_words in max_words_list:
        df_subset = sub_df[sub_df['max_words'] == max_words]
        if METRIC == "accuracy":
            if detector in ["roberta", "ghostbuster"]:
                threshold = .5
            else:
                threshold = get_threshold(df_subset, method="minimal-fp", allowed_error=0.05)
            score = accuracy_score(~df_subset["is_human"], df_subset["prediction"] >= threshold)
        elif METRIC == "f1-score":
            if detector in ["roberta", "ghostbuster"]:
                threshold = .5
            else:
                threshold = get_threshold(df_subset, method="minimal-fp", allowed_error=0.05)
            score = f1_score(~df_subset["is_human"], df_subset["prediction"] >= threshold)
        elif METRIC == "roc-auc":
            score = get_roc_auc(df_subset)

        if detector == "fast-detect-gpt":
            detector = "Fast-DetectGPT"
        elif detector == "detect-gpt":
            detector = "DetectGPT"
        elif detector == "intrinsic-dim":
            detector = "Intrinsic-Dim"
        elif detector == "ghostbuster":
            detector = "Ghostbuster"
        elif detector == "roberta":
            detector = "RoBERTa"

        scores.append({
            "detector": detector,
            "number of words": max_words if max_words > 0 else 300,
            METRIC: score,
        })

df = pd.DataFrame(scores)


def lineplot(data, **kwargs):
    for key, group in data.groupby("detector"):
        linestyle = "--" if group["inverted"].iloc[0] else "-"
        sns.lineplot(
            data=group,
            x="number of words",
            y=METRIC,
            linestyle=linestyle,
            **kwargs
        )


plt.figure(figsize=(10, 6))
g = sns.lineplot(df,
                 x="number of words",
                 y=METRIC,
                 hue="detector",
                 legend=True,
                 errorbar=None,
                 linewidth=5
                 )
g.legend(
    loc='upper center',
    bbox_to_anchor=(0.49, 1.25),  # center top, above axes
    ncol=4,  # spread horizontally
    frameon=False  # removes legend border
)
plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25, zorder=0)

plt.tight_layout()
plt.subplots_adjust(top=0.85, right=0.85, left=0.15)
g.set(xticks=[50, 100, 150, 200, 250, 300])

plt.xlabel("Maximum Number of Words", fontweight='bold')
plt.ylabel("ROC-AUC", fontweight='bold')
plt.savefig(f"plots/text_length_{METRIC}.pdf")
plt.show()
