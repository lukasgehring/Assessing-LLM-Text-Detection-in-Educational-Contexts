import re

from matplotlib import pyplot as plt
import seaborn as sns

from database.interface import get_full_table


def update_prompt_mode(row):
    if row.max_words != -1:
        row.prompt_mode = f"{row.prompt_mode}-{row.max_words}"

    row.prompt_mode = re.sub(r"rewrite-\d+", "rewrite-llm", row.prompt_mode)
    row.prompt_mode = re.sub(r"dipper-\d+", "dipper", row.prompt_mode)
    return row


df = get_full_table(database="../../database/database.db", table_name="experiments")
df.dropna(subset=["execution_time"], inplace=True)
df = df[df["dataset_id"] == 3]
df = df[df["text_author"] != "gpt-4o-mini-2024-07-18"]
# df = df[df["model"] != "detect-gpt"]
df = df.apply(update_prompt_mode, axis=1)
df = df[df["model_checkpoint"] != "detectors/RoBERTa/checkpoints/persuade/binary/checkpoint-24"]
df = df.sort_values(by="prompt_mode")

print(df.prompt_mode.unique())
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="prompt_mode", y="execution_time",
                hue="model",
                palette="ch:r=-.2,d=.3_r",
                linewidth=0,
                data=df, ax=ax)

ax.set(yscale="log")
plt.xticks(rotation=45)
# plt.show()
