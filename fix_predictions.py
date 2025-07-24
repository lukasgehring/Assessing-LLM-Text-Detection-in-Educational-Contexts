import sqlite3
import sys
from argparse import Namespace
from datetime import datetime

import pandas as pd
import torch

from detectors.RoBERTa.roberta_class import RoBERTa
from detectors.detect_gpt_based_detectors.detect_gpt import DetectGPT
from detectors.detect_gpt_based_detectors.fast_detect_gpt import FastDetectGPT
from detectors.ghostbuster.ghostbuster import Ghostbuster
from detectors.intrinsic_dim import IntrinsicDim
from utils.args import init_parser
from utils.logger import init_logger
from utils.truncate_text import apply_max_words


def process_text(text):
    return text.replace("\n", " ")


def update_predictions(args, detector, checkpoint=None):
    with sqlite3.connect('../database/database.db') as conn:
        df = pd.read_sql_query("""
            SELECT a.id, a.is_human, q.question, a.answer, prompt_mode
                FROM answers AS a
                JOIN questions AS q ON a.question_id = q.id
                JOIN datasets AS ds ON ds.id = q.dataset_id
                LEFT JOIN jobs AS j ON j.id = a.job_id
            WHERE a.modified_at >= '2025-07-20 00:00:00'
        """, conn)

    df['answer'] = df['answer'].apply(process_text)
    df.prompt_mode = df.prompt_mode.fillna('human')
    if args.max_words != -1:
        df = apply_max_words(df, args)
        df = df[df['prompt_mode'] == "human"]

    pred, ids = detector(df)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with sqlite3.connect('../database/database.db') as conn:
        cursor = conn.cursor()

        for id_value, prediction in zip(ids, pred):

            if checkpoint:
                cursor.execute(
                    f"""
                    UPDATE predictions
                    SET prediction = ?, modified_at = ?
                    WHERE answer_id = ? 
                    AND experiment_id IN (
                      SELECT e.id
                      FROM experiments e
                      WHERE e.model = '{args.model}'
                        AND e.model_checkpoint = '{checkpoint}'
                        AND e.max_words = {args.max_words}
                  )
                    """,
                    (prediction, now, id_value)
                )
            else:
                cursor.execute(
                    f"""
                                    UPDATE predictions
                                    SET prediction = ?, modified_at = ?
                                    WHERE answer_id = ? 
                                    AND experiment_id IN (
                                      SELECT e.id
                                      FROM experiments e
                                      WHERE e.model = '{args.model}'
                                        AND e.max_words = {args.max_words}
                                  )
                            """,
                    (prediction, now, id_value)
                )

        conn.commit()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # parse arguments
    default_args = init_parser()

    # detector_names = ['detect-gpt']
    checkpoints = ['detectors/RoBERTa/checkpoints/argument-annotated-essays/binary/checkpoint-123', 'detectors/RoBERTa/checkpoints/persuade/binary/checkpoint-24']
    max_words = [-1, 50, 100, 150, 200, 250]

    for checkpoint in checkpoints:

        parsed_args = args_copy = Namespace(**vars(default_args))

        parsed_args.checkpoint = checkpoint

        # if detector_name == 'fast-detect-gpt':
        #    parsed_args.base_model_name = 'EleutherAI/gpt-neo-2.7B'
        #    parsed_args.mask_filling_model_name = 'EleutherAI/gpt-j-6B'

        for max_word in max_words:
            parsed_args.model = 'roberta'  # detector_name
            parsed_args.max_words = max_word

            detector = RoBERTa(parsed_args)

            # if detector_name == 'fast-detect-gpt':
            #    detector = FastDetectGPT(parsed_args)
            # elif detector_name == 'detect-gpt':
            #    detector = DetectGPT(parsed_args)
            # elif detector_name == 'ghostbuster':
            #    detector = Ghostbuster(parsed_args)
            # elif detector_name == 'intrinsic-dim':
            #    detector = IntrinsicDim(parsed_args)
            # else:
            #    sys.exit("Unkonwn detector")

            update_predictions(args=parsed_args, detector=detector, checkpoint=checkpoint)
