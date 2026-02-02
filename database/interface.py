import json
import os
import sqlite3
import sys
import unittest
import warnings
from typing import List, Optional
from uuid import uuid4

import pandas as pd
from loguru import logger

import database

warnings.simplefilter("once", DeprecationWarning)


def init_db(db: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        # optional sinnvoll:
        conn.execute("PRAGMA synchronous = NORMAL;")


def process_text(text):
    return text.replace("\n", " ")


def connect_rw(db: str, timeout_s: int = 60) -> sqlite3.Connection:
    conn = sqlite3.connect(db, timeout=timeout_s)
    conn.execute(f"PRAGMA busy_timeout = {timeout_s * 1000};")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def get_answers_by_id(database: str, ids: List[int]) -> pd.DataFrame:
    """
    Load answers by id.
    :param database: Path to database
    :param ids: List of answer ids.
    :return: DataFrame with answers.
    """
    try:
        with connect_rw(database) as conn:
            placeholders = ', '.join('?' for _ in ids)
            df = pd.read_sql_query(f"""
                    SELECT a.id, a.is_human, q.question, a.answer
                    FROM answers AS a
                    JOIN questions AS q ON a.question_id = q.id
                    JOIN datasets AS ds ON ds.id = q.dataset_id
                    LEFT JOIN jobs AS j ON j.id = a.job_id
                    WHERE a.id IN ({placeholders})
                    ORDER BY a.id
            """, conn, params=ids)
    except sqlite3.Error as e:
        logger.error(e)
        sys.exit(1)

    return df


def get_answers(database: str, dataset: str, is_human: bool, prompt_mode: str = None,
                generative_model: str = None) -> pd.DataFrame:
    """
    Load human or llm answers from a database.

    :param database: Path to the database
    :param dataset: Name of the dataset
    :param is_human: Whether the answer is human
    :param prompt_mode: Prompt mode (if answer is not human)
    :param generative_model: Generative model (if answer is not human)
    :return: Dataframe with id, is_human, question and answer
    """

    # check if prompt_mode and generative_model are provided if answer is from llm
    if not is_human:
        if prompt_mode is None:
            logger.error("prompt_mode is undefined, please specify prompt_mode or set is_human to True")
            sys.exit(1)
        if generative_model is None:
            logger.error("model is undefined, please specify model or set is_human to True")
            sys.exit(1)

    # read from database
    try:
        with connect_rw(database) as conn:
            df = pd.read_sql_query(f"""
                    SELECT a.id, a.is_human, q.question, a.answer
                    FROM answers AS a
                    JOIN questions AS q ON a.question_id = q.id
                    JOIN datasets AS ds ON ds.id = q.dataset_id
                    LEFT JOIN jobs AS j ON j.id = a.job_id
                    WHERE ds.name = :name
                        AND (
                            (:is_human = 1 AND a.is_human)
                            OR
                            (:is_human = 0 AND j.prompt_mode = :prompt_mode AND j.model = :model)
                        )
                    ORDER BY a.id
            """, conn, params={
                'name': dataset,
                'is_human': is_human,
                'prompt_mode': prompt_mode,
                'model': generative_model
            })
    except sqlite3.Error as e:
        logger.error(e)
        sys.exit(1)

    # process answers
    df['answer'] = df['answer'].apply(process_text)

    return df


def get_all_answers(database: str):
    try:
        with connect_rw(database) as conn:
            df = pd.read_sql_query(f"""
                    SELECT a.id, a.is_human, a.answer
                    FROM answers AS a
                    ORDER BY a.id
            """, conn)
    except sqlite3.Error as e:
        logger.error(e)
        sys.exit(1)

    df['answer'] = df['answer'].apply(process_text)

    return df


def add_experiment(database: str, job_id: int, dataset: str, text_author: str, prompt_mode: str, model: str,
                   n_samples: int, max_words: int, cut_sentences: bool, use_cache: bool,
                   human_hash: str, llm_hash: str, seed: int, model_checkpoint: str) -> int:
    """
    Add experiment to database.

    :param database: Path to the database
    :param job_id: Job ID
    :param dataset: Name of the dataset
    :param text_author: Name of the text author (human or llm name)
    :param prompt_mode: Prompt mode (if answer is not human)
    :param model: Detector model
    :param n_samples: Number of samples
    :param max_words: Maximum number of words
    :param cut_sentences: Whether to cut sentences or not
    :param use_cache: Whether to use cache
    :param human_hash: Human hash
    :param llm_hash: LLM hash
    :param seed: Seed
    :param model_checkpoint: Checkpoint of the model
    :return: Experiment ID
    """

    data = {
        'job_id': job_id,
        'text_author': text_author,
        'prompt_mode': prompt_mode,
        'model': model,
        'n_samples': n_samples,
        'max_words': max_words if max_words is not None else -1,
        'cut_sentences': cut_sentences,
        'use_cache': use_cache,
        'human_hash': human_hash,
        'llm_hash': llm_hash,
        'seed': seed,
        'model_checkpoint': model_checkpoint
    }

    try:
        with connect_rw(database) as connection:

            cursor = connection.cursor()

            cursor.execute("SELECT ds.id FROM datasets AS ds WHERE ds.name = ?", (dataset,))
            dataset_id = cursor.fetchone()

            if dataset_id:
                data['dataset_id'] = dataset_id[0]
            else:
                logger.error(f"Dataset {dataset} unknown!")
                sys.exit(-1)

            # create query
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?'] * len(data))
            query = f"INSERT INTO experiments ({columns}) VALUES ({placeholders})"

            # execute query
            connection.execute("BEGIN IMMEDIATE;")
            cursor.execute(query, tuple(data.values()))
            experiment_id = cursor.lastrowid

            if experiment_id == 0:
                # experiment does already exist
                experiment_id = None
                logger.warning(f"Experiment with same settings already exists!")
            else:
                # commit changes
                connection.commit()

                logger.success(f"Successfully added experiment {experiment_id} to database.")
    except sqlite3.Error as e:
        logger.error(f"The following error occurred when creating experiment to database: {e}")
        experiment_id = None

    return experiment_id


def add_predictions(database: str, experiment_id: int, predictions: List, answer_ids: pd.Series) -> None:
    """
    Add predictions to database.

    :param database: Path to the database
    :param experiment_id: Experiment ID
    :param predictions: Predictions
    :param answer_ids: Answer IDs
    """

    # create DataFrame of predictions
    df = pd.DataFrame({"prediction": predictions})
    df['experiment_id'] = experiment_id
    df['answer_id'] = answer_ids.values

    # write to database
    try:
        with connect_rw(database) as connection:

            connection.execute("BEGIN IMMEDIATE;")
            df.to_sql('predictions', connection, if_exists='append', index=False, method='multi')
            connection.commit()

        logger.success(f"Successfully added predictions of experiment {experiment_id} to database.")
    except sqlite3.Error as e:
        logger.error(f"The following error occurred when adding predictions to database: {e}")
        df.to_csv(f'tmp_results_{experiment_id}.csv', sep=';')
        logger.info(f"Results written to 'tmp_results_{experiment_id}.csv'")


def get_predictions_by_answer_ids(database: str, answer_ids: List[int]) -> pd.DataFrame:
    try:
        with connect_rw(database) as conn:
            placeholders = ', '.join('?' for _ in answer_ids)
            df = pd.read_sql_query(f"""
                    SELECT p.id, p.experiment_id, p.answer_id, p.prediction, j.prompt_mode, e.model AS detector, j.model AS generative_model, e.max_words, ds.name, a.is_human, e.model_checkpoint
                    FROM predictions AS p
                    JOIN experiments AS e ON p.experiment_id = e.id
                    JOIN answers AS a ON p.answer_id = a.id
                    JOIN questions AS q ON q.id = a.question_id
                    JOIN datasets AS ds ON ds.id = q.dataset_id
                    LEFT JOIN jobs AS j ON j.id = a.job_id
                    WHERE p.answer_id IN ({placeholders}) AND e.max_words == -1
                    ORDER BY a.id
            """,
                                   conn,
                                   params=answer_ids,
                                   dtype={
                                       'prediction': 'float64',
                                       'is_human': 'bool'
                                   }
                                   )
    except sqlite3.Error as e:
        logger.error(e)
        sys.exit(1)

    # set human prompt mode
    df.loc[df['is_human'] == True, 'prompt_mode'] = 'human'

    return df


def get_predictions(database: str = "../../database/database.db", dataset: Optional[str] = None, prompt_mode: Optional[str] = None, generative_model: Optional[str] = None,
                    detector: Optional[str] = None, max_words: Optional[int] = -1,
                    is_human: Optional[bool] = None, model_checkpoint: Optional[str] = None, drop_gpt_zero: bool = True) -> pd.DataFrame:
    """
    Load predictions from a database.

    :param database: Path to the database
    :param dataset: Name of the dataset
    :param prompt_mode: Prompt mode (if answer is not human)
    :param generative_model: Generative model (if answer is not human)
    :param detector: Detector model
    :param max_words: Maximum number of words
    :param is_human: Whether to use human or llm (default False)
    :param model_checkpoint: Path to model checkpoint (optional)
    :return: DataFrame with id, experiment_id, answer_id, prediction, prompt_mode, detector, generative_model, max_words, dataset, is_human
    """
    try:
        with connect_rw(database) as conn:
            df = pd.read_sql(
                f"""
                        SELECT p.id, p.experiment_id, p.answer_id, p.prediction, e.prompt_mode, e.model AS detector, e.text_author AS generative_model, e.max_words, ds.name, a.is_human, e.model_checkpoint, q.id as question_id
                        FROM predictions AS p
                        JOIN experiments AS e ON p.experiment_id = e.id
                        JOIN answers AS a ON p.answer_id = a.id
                        JOIN questions AS q ON q.id = a.question_id
                        JOIN datasets AS ds ON ds.id = q.dataset_id
                        WHERE
                            -- If :dataset is provided, filter by dataset name; otherwise, allow all datasets
                            (:dataset IS NULL OR ds.name = :dataset) 

                            -- If :model_checkpoint is provided, filter by model_checkpoint; otherwise, allow all
                            AND (:model_checkpoint IS NULL OR e.model_checkpoint = :model_checkpoint) 

                            -- If :detector is provided, filter by model; otherwise, allow all models
                            AND (:detector IS NULL OR e.model = :detector) 

                            -- If :max_words is provided, filter by max_words; otherwise, allow all values
                            AND (:max_words IS NULL OR e.max_words = :max_words)
                            AND (

                                -- If :is_human is NULL (not provided), allow both human and non-human answers
                                -- - Allow all human-generated answers
                                -- - For non-human answers, still apply prompt_mode and generative_model filters
                                (:is_human IS NULL 
                                    AND (a.is_human 
                                         OR ((:prompt_mode IS NULL OR e.prompt_mode = :prompt_mode) 
                                             AND (:generative_model IS NULL OR e.text_author = :generative_model)
                                         )
                                    )
                                )

                                -- If :is_human is explicitly 1, only allow human-generated answers
                                OR (:is_human = 1 AND a.is_human)
                                -- If :is_human is explicitly 0, only allow non-human-generated answers 
                                -- and apply filters for prompt_mode and generative_model
                                OR (:is_human = 0
                                    AND a.is_human = 0
                                    AND (:prompt_mode IS NULL OR e.prompt_mode = :prompt_mode)
                                    AND (:generative_model IS NULL OR e.text_author = :generative_model)
                                    )
                                );
                """,
                conn,
                params={
                    'dataset': dataset,
                    'prompt_mode': prompt_mode,
                    'detector': detector,
                    'max_words': max_words,
                    'generative_model': generative_model,
                    'is_human': is_human,
                    'model_checkpoint': model_checkpoint,
                },
                dtype={
                    'prediction': 'float64',
                    'is_human': 'bool'
                })
    except sqlite3.Error as e:
        logger.error(f"The following error occurred when reading predictions from database: {e}")
        sys.exit(1)

    if drop_gpt_zero:
        df = df[df['detector'] != 'gpt-zero']

    # set human prompt mode
    df.loc[df['is_human'] == True, 'prompt_mode'] = 'human'

    return df


def add_execution_time(database: str, experiment_id: int, execution_time: int) -> None:
    try:
        with connect_rw(database) as conn:
            conn.execute("BEGIN IMMEDIATE;")

            cursor = conn.cursor()
            cursor.execute("UPDATE experiments SET execution_time = ? WHERE id = ?", (execution_time, experiment_id))
            conn.commit()
            logger.success(f"Successfully updated execution time {execution_time} to experiment {experiment_id}.")

    except sqlite3.Error as e:
        logger.error(f"The following error occurred when updating execution time in database: {e}")
        sys.exit(1)


def get_human_meta_data(database: str, answer_ids: pd.Series):
    try:
        with connect_rw(database) as conn:
            ids = answer_ids.tolist()
            query = f"SELECT * FROM human_meta WHERE answer_id IN ({','.join('?' * len(ids))})"

            df = pd.read_sql_query(query, conn, params=ids)

            return df
    except sqlite3.Error as e:
        logger.error(f"The following error occurred when reading human meta data from database: {e}")
        sys.exit(1)


def get_full_table(database: str, table_name: str) -> pd.DataFrame:
    try:
        with connect_rw(database) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except sqlite3.Error as e:
        logger.error(f"The following error occurred when reading table {table_name} from database: {e}")

    return df


def add_question_metadata_to_df(database: str, df: pd.DataFrame) -> pd.DataFrame:
    try:
        with connect_rw(database) as conn:
            ids = df.question_id.tolist()
            query = f"SELECT id as question_id, module, discipline, discipline_group, course FROM questions WHERE id IN ({','.join('?' * len(ids))})"

            meta_data = pd.read_sql_query(query, conn, params=ids)

            df = df.merge(meta_data, on='question_id', how='left')

            return df
    except sqlite3.Error as e:
        logger.error(f"The following error occurred when reading human meta data from database: {e}")
        sys.exit(1)


class Test(unittest.TestCase):

    def setUp(self):
        self.database = "../database/database.db"

    def test_all_predictions(self):
        self.assertEqual(158408, len(get_predictions(database=self.database, max_words=None)))

    def test_generative_model(self):
        self.assertEqual(58286, len(get_predictions(database=self.database, generative_model="gpt-4o-mini-2024-07-18", is_human=False, max_words=None)))
        self.assertEqual(89468, len(get_predictions(database=self.database, generative_model="gpt-4o-mini-2024-07-18", max_words=None)))
        self.assertEqual(89542, len(get_predictions(database=self.database, generative_model="meta-llama/Llama-3.3-70B-Instruct", max_words=None)))
        self.assertEqual(41762, len(get_predictions(database=self.database, generative_model="dipper", max_words=None)))

    def test_max_words(self):
        self.assertEqual(32491, len(get_predictions(database=self.database, generative_model="gpt-4o-mini-2024-07-18", is_human=False, max_words=-1)))
        self.assertEqual(5468, len(get_predictions(database=self.database, is_human=True, max_words=-1)))
        self.assertEqual(37959, len(get_predictions(database=self.database, generative_model="gpt-4o-mini-2024-07-18", max_words=-1)))

        self.assertEqual(5169, len(get_predictions(database=self.database, is_human=True, max_words=250)))
        self.assertEqual(0, len(get_predictions(database=self.database, is_human=True, max_words=500)))

    def test_detector(self):
        self.assertEqual(26893, len(get_predictions(database=self.database, detector="detect-gpt", max_words=None)))
        self.assertEqual(54622, len(get_predictions(database=self.database, detector="roberta", max_words=None)))

    def test_prompt_mode(self):
        self.assertEqual(16105, len(get_predictions(database=self.database, prompt_mode="summary")))

    def test_dataset(self):
        self.assertEqual(39013, len(get_predictions(database=self.database, dataset="BAWE")))


if __name__ == '__main__':
    unittest.main()
