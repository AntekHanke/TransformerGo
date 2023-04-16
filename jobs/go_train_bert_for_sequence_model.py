from typing import Type, List

import pandas as pd

# import evaluate

from transformers.integrations import NeptuneCallback

from data_processing.go_data_generator import GoGamesDataGenerator, GoDataProvider
from data_processing.go_tokenizer import GoTokenizer
from utils.data_utils import immutable_boards_to_img
from jobs.core import Job
from transformers import (
    Trainer,
    BartConfig,
    TrainingArguments,
    BertForSequenceClassification,
)

from metric_logging import log_param, source_files_register, pytorch_callback_loggers, log_object
from utils.global_params_handler import GlobalParamsHandler

source_files_register.register(__file__)

# @TODO - like for value -> logging of samples used for training
# @TODO - train model and write value tests for it
# @TODO - compare with Stockfish -> take 20 states, sort them by value and see if they are in same order as Stockfish


class GoTrainBertForSequenceModel(Job):
    def __init__(
        self,
        go_database_cls: Type[GoDataProvider],
        model_config_cls: Type[BartConfig] = None,
        training_args_cls: Type[TrainingArguments] = None,
        output_dir: str = None,
    ):

        if GlobalParamsHandler().out_dir is not None:
            output_dir = GlobalParamsHandler().out_dir

        log_param("output_dir", output_dir)

        go_database = go_database_cls()
        if isinstance(go_database, GoGamesDataGenerator):
            go_database.create_data()

        # log data in neptune
        self._log_data_to_neptune(go_database)

        self.model_config = model_config_cls()
        self.training_args = training_args_cls(output_dir=output_dir + "/out")

        if GlobalParamsHandler().learning_rate is not None:
            self.training_args.learning_rate = GlobalParamsHandler().learning_rate

        self.model = BertForSequenceClassification(self.model_config)

        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        log_param("number of parameters", self.self.pytorch_total_params)

        log_param("real learning rate", self.training_args.learning_rate)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=go_database.get_train_set_generator(),
            eval_dataset=go_database.get_eval_set_generator(),
            # compute_metrics=compute_metrics,
        )

        for callback_logger in pytorch_callback_loggers:
            self.trainer.add_callback(callback_logger.get_pytorch_callback())
        self.trainer.pop_callback(NeptuneCallback)
        self.save_model_path = output_dir + "/final_model"
        log_param("Save model path", self.save_model_path)

    def _log_data_to_neptune(self, chess_database: GoDataProvider) -> None:

        def _dict_to_pandas(data_dict: dict, max_data_to_log: int = 100) -> pd.DataFrame:
            dfs = []
            for id, value in enumerate(data_dict.values()):
                if id >= max_data_to_log:
                    break
                new_value = value.copy()
                new_value['labels'], new_value['input_ids'] = [new_value['labels']], [new_value['input_ids']]
                df = pd.DataFrame.from_dict(new_value, orient="columns")
                dfs.append(df)
            return pd.concat(dfs)

        def _detokenize_df_row(tokens: List[int], label: List[str]):
            return GoTokenizer().decode_boards_active_player(tokens)

        def _log_figures_for_dataset(go_database: GoDataProvider, dataset_name: str) -> None:
            attr = getattr(go_database, dataset_name)
            df = _dict_to_pandas(attr)
            df['detokenized_ids'] = df.apply(lambda row: _detokenize_df_row(row['input_ids'], row['labels']), axis=1)
            for id, row in enumerate(df[['detokenized_ids', 'labels']].iterrows()):
                #fig = immutable_boards_to_img([row[1]['detokenized_ids']], [row[1]['labels']])
                # log_object(f'{dataset_name}_sample_img', fig)
                # log_object(f'{dataset_name}_sample', f"{id}\t{row[1]['detokenized_ids'].board}")
                log_object(f'{dataset_name}_sample', row)

        # chess_database_for_logging
        _log_figures_for_dataset(chess_database, 'data_train')
        _log_figures_for_dataset(chess_database, 'data_eval')

    def execute(self) -> None:
        self.trainer.train()
        print(f"Saving model at {self.save_model_path}")
        self.trainer.save_model(self.save_model_path)


if __name__ == '__main__':

    generator = GoSimpleGamesDataGeneratorTokenizedAlwaysBlack(sgf_files='val.txt',save_data_every_n_games=990,p_sample=1,max_games=9950,train_eval_split=0.95,save_path_to_eval_set='tokenized_data\\eval',save_path_to_train_set='tokenized_data\\train')
    generator.create_data()