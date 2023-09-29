from argparse import ArgumentParser
from pcrl.utils.logging_utils import Tracker
from pcrl.utils.training_utils import OnPolicyTrainer
import os
import sys
sys.path.append('.')
import random
import yaml
import numpy as np
import torch

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(
    config_path: str,
    project_name: str,
    experiment_name: str,
    base_path_to_store_results: str,
    entity_name: str,
    log_to_wandb: bool, 
    seed: int,
):

    # load the config file
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    set_seed_everywhere(seed)

    # load tracker
    tracker = Tracker(
        base_path_to_store_results,
        config,
        project_name,
        experiment_name,
        entity_name,
        log_to_wandb,
    )

    trainer = OnPolicyTrainer(
        gen_config=config["gen_model"],
        datapool_config=config["datapool"],
        reward_config=config["reward_fn"],
        env_config=config["env"],
        alg_config=config["alg"],
        train_eval_config=config["train_evaluation"],
        tracker=tracker,
    )
    trainer.train_and_eval()


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune LM to generate controlled text")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--project_name", type=str, help="WANDB project name", default="pcrl_exps"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="WANDB experiment name",
        default="llama2_2023",
    )
    parser.add_argument(
        "--entity_name", type=str, help="WANDB entity name",
    )
    parser.add_argument(
        "--base_path_to_store_results",
        type=str,
        help="Base path to store experiment results",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--log_to_wandb", action="store_true", help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--seed", type=int, help="random seed to use", default=2023
    )
    args = parser.parse_args()

    main(
        args.config_path,
        args.project_name,
        args.experiment_name,
        args.base_path_to_store_results,
        args.entity_name,
        args.log_to_wandb,
        args.seed,
    )
