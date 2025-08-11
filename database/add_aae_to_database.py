import sqlite3

import requests
import zipfile
import io
import pandas as pd

url = 'https://tudatalib.ulb.tu-darmstadt.de/bitstreams/1ae1718d-7e65-42ba-9e84-dbf52fe92f56/download'

response = requests.get(url)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall("Argument-Annotated-Essays")

with zipfile.ZipFile("Argument-Annotated-Essays/ArgumentAnnotatedEssays-2.0/brat-project-final.zip") as zip_ref:
    zip_ref.extractall("Argument-Annotated-Essays/ArgumentAnnotatedEssays-2.0/brat-project-final")

print('-' * 20)
print("Download successful!\nPlease read the license.pdf under database/Argument-Annotated-Essays/ArgumentAnnotatedEssays-2.0/license.pdf")
print('-' * 20)

df = pd.read_csv("Argument-Annotated-Essays/ArgumentAnnotatedEssays-2.0/prompts.csv", encoding='ISO-8859-1', delimiter=';')
prompts = df.drop("ESSAY", axis=1)
files = df.drop("PROMPT", axis=1)
prompts.columns = ['question']
prompts = prompts.drop_duplicates(subset=['question']).reset_index(drop=True)
prompts['question'] = prompts['question'].str.replace('Ê', ' ', regex=False)
prompts['question'] = prompts['question'].str.strip()

with sqlite3.connect('database.db') as connection:
    cursor = connection.cursor()
    for idx, row in prompts.iterrows():
        cursor.execute("""
            UPDATE questions 
            SET question = ? 
            WHERE id = ?;
        """, (row['question'], idx + 1))
    connection.commit()

essays = []

for file in files['ESSAY']:
    with open(f"Argument-Annotated-Essays/ArgumentAnnotatedEssays-2.0/brat-project-final/brat-project-final/{file}", "r") as f:
        essay = ''.join(f.readlines()[2:])  # remove the heading of the essay
        essays.append(essay.rstrip())  # remove \n at the end of an essay

essays = pd.DataFrame({"answer": essays})

with sqlite3.connect('database.db') as connection:
    cursor = connection.cursor()
    for idx, row in essays.iterrows():
        cursor.execute("""
            UPDATE answers 
            SET answer = ? 
            WHERE id = ?;
        """, (row['answer'], idx + 1))
    connection.commit()

print("Argument-Annotated-Essays corpus successfully updated. You can now use the database!")
