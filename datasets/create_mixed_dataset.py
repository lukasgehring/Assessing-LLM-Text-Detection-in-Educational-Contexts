import re
import sqlite3
import sys

import pandas as pd

from database.interface import get_answers_by_id


def sample_questions(x):
    if x.name == 2:
        x = x[x.is_original == True]
        return x

    else:
        return x.sample(n=40, random_state=42)


def load_data():
    with sqlite3.connect('../../database/database.db') as conn:
        questions = pd.read_sql_query("""
        SELECT * FROM questions
        """, conn)

        answers = pd.read_sql_query("""
            SELECT answers.*, jobs.prompt_mode, jobs.model, questions.dataset_id, questions.rewrite_from FROM answers
            JOIN questions ON questions.id = answers.question_id
            LEFT JOIN jobs ON answers.job_id = jobs.id
            
            """, conn)
    return questions, answers


def rename_modes(df):
    df.loc[:, 'prompt_mode'] = df['prompt_mode'].replace(r"^rewrite-\d+$", "rewrite-llm",
                                                         regex=True)
    df.loc[:, 'prompt_mode'] = df['prompt_mode'].replace(r"^dipper-\d+$", "dipper",
                                                         regex=True)
    df.loc[:, 'prompt_mode'] = df['prompt_mode'].fillna('human')
    df.loc[:, 'model'] = df['model'].fillna('human')


def sample_answers(df):
    total_words = 0
    res = {}
    for mode, group in df.groupby('prompt_mode'):
        if mode == "task+resource":
            continue
        sampled = group.groupby(['dataset_id', 'question_id', 'model'], group_keys=False).apply(
            lambda x: x.sample(n=1, random_state=42), include_groups=False)
        sampled = group[group['id'].isin(sampled['id'])].sort_values('id')

        for mode2, group2 in sampled.groupby('model'):
            num_words = 0
            for text in sampled.answer:
                words = re.findall(r'\b\w+\b', text)  # erkennt Wörter
                num_words += len(words)
            total_words += num_words

            sampled2 = group2[group2['id'].isin(sampled['id'])].sort_values('id')

            res[f"{mode}-{mode2}"] = sampled2
    print(f"Total words: {total_words}")
    return res


def rest():
    filtered_answers = answers[answers['question_id'].isin(sampled_questions['id'])]
    filtered_answers.loc[:, 'prompt_mode'] = filtered_answers['prompt_mode'].replace(r"^rewrite-\d+$", "rewrite-llm",
                                                                                     regex=True)
    filtered_answers.loc[:, 'prompt_mode'] = filtered_answers['prompt_mode'].replace(r"^dipper-\d+$", "dipper",
                                                                                     regex=True)
    filtered_answers.loc[:, 'prompt_mode'] = filtered_answers['prompt_mode'].fillna('human')
    filtered_answers.loc[:, 'model'] = filtered_answers['model'].fillna('human')
    dfs_by_prompt_mode = {}

    total_words = 0
    for mode, group in filtered_answers.groupby('prompt_mode'):
        if mode == "task+resource":
            continue
        sampled = group.groupby(['dataset_id', 'question_id', 'model'], group_keys=False).apply(
            lambda x: x.sample(n=1, random_state=42), include_groups=False)
        sampled = group[group['id'].isin(sampled['id'])].sort_values('id')

        for mode2, group2 in sampled.groupby('model'):
            num_words = 0
            for text in sampled.answer:
                words = re.findall(r'\b\w+\b', text)  # erkennt Wörter
                num_words += len(words)
            total_words += num_words

            sampled2 = group2[group2['id'].isin(sampled['id'])].sort_values('id')

            dfs_by_prompt_mode[f"{mode}-{mode2}"] = sampled2


if __name__ == '__main__':
    questions, answers = load_data()

    sampled_questions = questions.groupby(['dataset_id'], group_keys=False).apply(sample_questions,
                                                                                  include_groups=False)
    filtered_answers = answers[answers['question_id'].isin(sampled_questions['id'])]
    rename_modes(filtered_answers)

    filtered_answers = filtered_answers[
        filtered_answers['prompt_mode'].isin(['human', 'task', 'improve-human', 'dipper'])]
    res = sample_answers(filtered_answers)

    dataset = pd.concat(res.values(), ignore_index=True)

    dataset.sort_values('id', inplace=True)

    dataset.id.to_csv("mixed_dataset_ids.txt", index=False, header=False)

    df = get_answers_by_id("../../database/database.db", dataset.id.tolist())
