from jobs.core import Job


class RetokenizationJob(Job):
    def __init__(self, source_directory: str, target_directory: str):
        self.source_directory = source_directory
        self.target_directory = target_directory

    def execute(self):
        from metric_logging import log_object, log_value_without_step
        import os
        import pickle
        from data_processing.chess_tokenizer import ChessTokenizerBoard, ChessTokenizerPiece

        for i, filename in enumerate(os.listdir(self.source_directory)):
            log_value_without_step("File number", i)
            if filename[-4:] == ".pkl":
                source_path = os.path.join(self.source_directory, filename)
                target_path = os.path.join(self.target_directory, filename)
                if os.path.isfile(source_path):
                    with open(source_path, "rb") as pickled_df:
                        df = pickle.load(pickled_df)
                        if len(df) == 0:
                            log_object("Empty file", source_path)
                            continue
                        df["input_ids"] = df["input_ids"].apply(
                            lambda x: ChessTokenizerPiece.encode_immutable_board(ChessTokenizerBoard.decode_board(x))
                        )
                        df["labels"] = df["labels"].apply(
                            lambda x: ChessTokenizerPiece.encode_immutable_board(ChessTokenizerBoard.decode_board(x))
                        )
                    df.to_pickle(target_path)
                    if os.path.isfile(target_path):
                        print(f"Saved {target_path}")
