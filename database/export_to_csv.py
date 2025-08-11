import pandas as pd
import sqlite3


def export_answers(db_name):
    csv_file_name = "../dataset/essays.csv"
    conn = sqlite3.connect(db_name)

    df = pd.read_sql_query(
        f"""
    SELECT 
        answers.*, 
        datasets.name AS dataset,
        jobs.prompt_mode AS contribution_level,
        jobs.model AS text_author
    FROM 
        answers 
    JOIN 
        questions ON questions.id = answers.question_id 
    JOIN 
        datasets ON questions.dataset_id = datasets.id 
    LEFT JOIN 
        jobs ON answers.job_id = jobs.id
    """,
        conn)

    df.drop(columns=['created_at', 'modified_at', 'job_id', 'is_human'], inplace=True)
    df.fillna('human', inplace=True)

    print(df['contribution_level'].unique())

    df['contribution_level'] = df['contribution_level'].replace("improve-human", "Improved-Human")
    df['contribution_level'] = df['contribution_level'].replace(r"^rewrite-\d+$", "Rewrite-LLM", regex=True)
    df['contribution_level'] = df['contribution_level'].replace(r"^dipper-\d+$", "Humanize", regex=True)
    df['contribution_level'] = df['contribution_level'].replace("rewrite-human", "Rewrite-Human")
    df['contribution_level'] = df['contribution_level'].replace("task", "Task")
    df['contribution_level'] = df['contribution_level'].replace("summary", "Summary")
    df['contribution_level'] = df['contribution_level'].replace("task+summary", "Task+Summary")
    df['contribution_level'] = df['contribution_level'].replace("human", "Human")

    df.to_csv(csv_file_name, index=False, encoding='utf-8')

    conn.close()
    print(f'Daten aus der Tabelle answers wurden erfolgreich in {csv_file_name} exportiert.')


def export_questions(db_name):
    csv_file_name = "../dataset/questions.csv"
    conn = sqlite3.connect(db_name)

    df = pd.read_sql_query(
        f"""
    SELECT 
        *
    FROM 
        questions 
    """,
        conn)

    df.drop(columns=['created_at', 'modified_at', 'dataset_id'], inplace=True)

    df.to_csv(csv_file_name, index=False, encoding='utf-8')

    conn.close()
    print(f'Daten aus der Tabelle answers wurden erfolgreich in {csv_file_name} exportiert.')


db_name = '../database/database.db'

print("MAKE SURE TO RUN add_aae_to_database.py before!")

export_answers(db_name)
export_questions(db_name)
