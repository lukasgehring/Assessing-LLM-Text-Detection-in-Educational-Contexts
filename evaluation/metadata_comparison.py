import sqlite3
import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from database.interface import get_predictions, add_question_metadata_to_df
from evaluation.utils import select_best_roberta_checkpoint, set_label, remove_job_id_from_prompt_mode, \
    filter_and_get_roc_auc, get_roc_auc


def load_info():
    info = pd.read_excel("../../HumanWrittenTextDatasets/raw/BAWE/raw/BAWE.xls")
    info['title'] = info['title'].str.capitalize()
    return info.drop_duplicates(subset='title')


def fetch_questions(conn, ids):
    placeholders = ','.join('?' * len(ids))
    query = f"SELECT id, question FROM questions WHERE id IN ({placeholders})"
    return pd.read_sql_query(query, conn, params=ids)


def select_metadata(info, questions):
    metas = []
    for _, row in questions.iterrows():
        match = info[info['title'] == row['question']]
        if match.empty:
            raise ValueError(f"No metadata found for question: {row['question']}")
        if len(match) != 1:
            raise ValueError(f"Multiple metadata entries for: {row['question']}")

        meta_entry = match.iloc[0].drop(['title', 'id'], errors='ignore').to_dict()
        meta_entry['question_id'] = row['id']
        metas.append(meta_entry)
    return pd.DataFrame(metas)


def build_data():
    df = get_predictions(dataset="BAWE", max_words=-1)
    info = load_info()

    with sqlite3.connect("../../database/database.db") as conn:
        questions = fetch_questions(conn, df['question_id'].tolist())

    metadata = select_metadata(info, questions)
    df = df.merge(metadata, on='question_id', how='left')
    df.rename(columns={'disciplinary group': 'discipline_group'}, inplace=True)

    return df


def compare_meta(df, key):
    results = []
    for (detector, meta_info), sub_df in df.groupby(['detector', key]):
        results.append({
            "detector": detector,
            "key": meta_info,
            "prompt_mode": "all",
            "counts": len(sub_df.index),
            "roc_auc": get_roc_auc(sub_df)
        })
    df = pd.DataFrame(results)

    df_grouped = df.groupby(["key"])['roc_auc'].agg(['mean', 'std']).reset_index()
    df_grouped['ROC-AUC (Mean ± Std)'] = df_grouped.apply(
        lambda x: f"{x['mean']:.3f} ± {x['std']:.2f}", axis=1
    )
    df_grouped.sort_values(by=['mean'], ascending=False, inplace=True)
    df_grouped.drop(columns=["mean", "std"], inplace=True)
    df_grouped = df_grouped.reset_index(drop=True)

    df_grouped['counts'] = df.counts

    return df_grouped


df = build_data()
df = select_best_roberta_checkpoint(df)
set_label(df)
remove_job_id_from_prompt_mode(df)

print(compare_meta(df, 'L1'))

sys.exit()

results = []
for (detector, discipline, discipline_group), sub_df in df.groupby(['detector', 'discipline', 'discipline_group']):
    results.append({
        "detector": detector,
        "discipline_group": discipline_group,
        "discipline": discipline,
        "prompt_mode": "all",
        "roc_auc": get_roc_auc(sub_df)
    })
df = pd.DataFrame(results)

df_grouped = df.groupby(["discipline", "discipline_group"])['roc_auc'].agg(['mean', 'std']).reset_index()
df_grouped['ROC-AUC (Mean ± Std)'] = df_grouped.apply(
    lambda x: f"{x['mean']:.3f} ± {x['std']:.2f}", axis=1
)
df_grouped.sort_values(by=['discipline_group', 'mean'], ascending=False, inplace=True)
df_grouped.drop(columns=["mean", "std"], inplace=True)
df_grouped = df_grouped.reset_index(drop=True)

with sqlite3.connect("../../database/database.db") as conn:
    counts = pd.read_sql("""
        select discipline_group, discipline,count(*) as c
        from questions
        where dataset_id = 3
        group by discipline_group, discipline
    """, conn)

# Schritt 1: Original-Disziplin-Werte gruppieren
df_grouped = df.groupby(["discipline", "discipline_group"])['roc_auc'].agg(['mean', 'std']).reset_index()
df_grouped['ROC-AUC (Mean ± Std)'] = df_grouped.apply(
    lambda x: f"{x['mean']:.3f} ± {x['std']:.2f}", axis=1
)

df_grouped.sort_values(by=['discipline_group', 'mean'], ascending=False, inplace=True)

# Schritt 2: Group-Mittelwert pro discipline_group berechnen
df_group_stats = df.groupby("discipline_group")['roc_auc'].agg(['mean', 'std']).reset_index()
df_group_stats['Group Mean ± Std'] = df_group_stats.apply(
    lambda x: f"{x['mean']:.3f} ± {x['std']:.2f}", axis=1
)

# Schritt 3: Mergen anhand von discipline_group
df_merged = pd.merge(df_grouped, df_group_stats[['discipline_group', 'Group Mean ± Std']],
                     on='discipline_group', how='left')

df_merged['dis_cout'] = 111
df_merged['group_cout'] = 999

for _, item in counts.iterrows():
    df_merged.loc[(df_merged.discipline_group == item.discipline_group) & (
                df_merged.discipline == item.discipline), 'dis_cout'] = item.c

# Schritt 4: Aufräumen
df_merged = df_merged[
    ["discipline", "ROC-AUC (Mean ± Std)", "dis_cout", "discipline_group", "Group Mean ± Std", "group_cout"]]

output = df_merged.to_latex(
    index=False,
    header=True,
    column_format="lcclcc",
    escape=False
)

# Leere Liste für die LaTeX-Zeilen
latex_rows = []

# Gruppieren nach discipline_group
for group, subdf in df_merged.groupby("discipline_group"):
    n_rows = len(subdf)
    group_mean = subdf["Group Mean ± Std"].iloc[0]
    group_count = subdf["group_cout"].iloc[0]

    for i, row in subdf.iterrows():
        if i == subdf.index[0]:
            latex_rows.append(
                f"\\multirow{{{n_rows}}}{{*}}{{{group}}} & "
                f"\\multirow{{{n_rows}}}{{*}}{{{group_mean}}} & "
                f"\\multirow{{{n_rows}}}{{*}}{{{group_count}}} & "
                f"{row['discipline']} & {row['ROC-AUC (Mean ± Std)']} & {row['dis_cout']} \\\\"
            )
        else:
            latex_rows.append(
                f" &  &  & {row['discipline']} & {row['ROC-AUC (Mean ± Std)']} & {row['dis_cout']} \\\\"
            )

# Header und Footer für die Tabelle
header = (
    "\\begin{tabular}{lcccccc}\n"
    "\\toprule\n"
    "discipline\\_group & Group Mean ± Std & group\\_cout & discipline & ROC-AUC (Mean ± Std) & dis\\_cout \\\\\n"
    "\\midrule"
)

footer = "\\bottomrule\n\\end{tabular}"

# Tabelle zusammensetzen
latex_table = header + "\n" + "\n".join(latex_rows) + "\n" + footer

# Ausgabe
print(latex_table)
