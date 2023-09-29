from collections import defaultdict
from typing import Dict, Any, List
from transformers import AutoModel
from rich.logging import RichHandler
import os
import json
import jsonlines
import wandb
import pandas as pd
import logging
import copy
import random

class Tracker:
    def __init__(self,
                 base_path_to_store_results: str,
                 run_config: Dict[str, Any],
                 project_name: str,
                 experiment_name: str,
                 entity_name: str = None,
                 wandb_log: bool = False,
                 log_level: int = logging.DEBUG,
        ):
        self._log_level = log_level
        self._base_path_to_store_results = base_path_to_store_results
        self._config = run_config
        self._experiment_name = experiment_name
        self._project_name = project_name
        self._entity_name = entity_name
        self._wandb_log = wandb_log
        self._init()

    def _init(self):
        # create a folder
        self._run_path = os.path.join(
            self._base_path_to_store_results,
            self._project_name,
            self._experiment_name)
        os.makedirs(self._run_path, exist_ok=True)

        # store also the config into it
        config_path = os.path.join(self._run_path, "config.json")
        with open(config_path, "w") as fp:
            json.dump(self._config, fp)

        # init logger
        log_path = os.path.join(self._run_path, "log.txt")
        logging.basicConfig(
            level=self._log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                RichHandler()
            ]
        )

        # init wandb
        if self._wandb_log:
            self._wandb_run = wandb.init(
                entity=self._entity_name,
                project=self._project_name,
                name=self._experiment_name,
                config=self._config,
                save_code=True,
            )
            self._wandb_run.log_code(".")

    def log_predictions(self, epoch: int,
                        split_name: str,
                        predictions: List[Dict]):
        # log them per epoch in a separate file as they can get huge
        prediction_file_at_epoch = os.path.join(
            self._run_path, f"epoch_{epoch}_{split_name}_split_predictions.json")
        with open(prediction_file_at_epoch, "w") as fp:
            json.dump(predictions, fp)

        # randomly display few predictions for logging
        predictions_ = copy.deepcopy(predictions)
        random.shuffle(predictions_)
        logging.info(f"Split {split_name} predictions")
        for pred in predictions_[:10]:
            logging.info(pred)

        # for wandb logging, we create a table consisting of predictions
        # we can create one table per split per epoch
        if self._wandb_log and len(predictions) > 0:

            def to_df(predictions):
                columns = predictions[0].keys()
                data_by_column = defaultdict(list)
                for item in predictions:
                    for column in columns:
                        data_by_column[column].append(item.get(column, ""))
                data_df = pd.DataFrame(data_by_column)
                return data_df

            predictions_as_df = to_df(predictions)
            predictions_table_at_epoch = wandb.Table(data=predictions_as_df)
            self._wandb_run.log({
                f"{split_name}_predictions_at_epoch_{epoch}": predictions_table_at_epoch})

    def log_metrics(self, epoch: int,
                    split_name: str,
                    metrics_dict: Dict[str, float]):
        # for each split, one file
        metric_file_per_split = os.path.join(
            self._run_path, f"{split_name}_split_metrics.jsonl")
        metrics_dict_ = {
            "epoch": epoch,
            "metrics": metrics_dict
        }
        with jsonlines.open(metric_file_per_split, "a") as writer:
            writer.write(metrics_dict_)

        # log to wandb
        if self._wandb_log:
            metric_dict_ = {
                f"eval/{split_name}/{metric_key}": value for metric_key, value in metrics_dict.items()}
            metric_dict_["epoch"] = epoch
            wandb.log(metric_dict_)

        # logger
        logging.info(f"{split_name} metrics: {metrics_dict_}")

    def log_rollout_infos(self, rollout_info: Dict[str, float]):
        logging.info(f"Rollout Info: {rollout_info}")
        rollout_info_file = os.path.join(
            self._run_path, "rollout_info.jsonl")
        with jsonlines.open(rollout_info_file, mode="a") as writer:
            writer.write(rollout_info)

        # log to wandb
        if self._wandb_log:
            wandb.log(rollout_info)

    def log_training_infos(self, training_info: Dict[str, float]):
        logging.info(f"Training Info: {training_info}")
        training_info_file = os.path.join(
            self._run_path, "training_info.jsonl")
        with jsonlines.open(training_info_file, mode="a") as writer:
            writer.write(training_info)

        # log to wandb
        if self._wandb_log:
            wandb.log(training_info)

    def done(self):
        if self._wandb_log:
            wandb.finish()

    def save_auto_model(self, model: AutoModel):
        model_path = os.path.join(self._run_path, "model")
        model.save_pretrained(model_path)

    @property
    def checkpoint_base_path(self):
        return os.path.join(self._run_path, "checkpoints")

    def log_info(self, msg: str):
        logging.info(msg)
