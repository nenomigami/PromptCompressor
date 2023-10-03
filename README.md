# Prompt Compression with Reinforcement Learning

## Introduction
This repository contains the code and datasets for the study on "[Discrete Prompt Compression with Reinforcement Learning](https://arxiv.org/abs/2308.08758)" The project aims to explore the practical way to compress prompts in instruction-tuned models with the application of reinforcement learning, thereby improving efficiency and performance.

## Requirements

- Python 3.9 
- PyTorch 1.13.1
- Other dependencies listed in `requirements.txt`

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nenomigami/PromptCompressor.git
2. Install the required packages:
    ```
    cd PromptCompressor
    pip install -r requirements.txt
    ```

## Datasets
The dataset used in this study includes the Alpaca+ from the [Mu et al. (2023) repository](https://github.com/jayelm/gisting). We have directly utilized this dataset to conduct our experiments on prompt compression. The Alpaca+ data is located in `data/alpaca_plus`. 

## Training
The training process consists of two main steps:

### 1. Instruction Fine-Tuning

First, we fine-tune the existing foundation models (gpt2-xl, flan-t5-xl) on the alpaca+ dataset. If you want to apply it to your own project and have a preferred instruction-tuned model, you can proceed with that as well and skip this section. 

For gpt2-xl:
```
python script/finetune_gpt2.py
```
For flan-t5-xl
```
python script/finetune_flan-t5-xl.py
```

### 2. Training PCRL 
After fine-tuning the models, we proceed to train the PCRL using the following command:

```
python train_pcrl.py --config_path configs/gpt2.yml --log_to_wandb --seed=myseed --experiment_name=my_experiment
```

## Evaluation

The evaluation process involves executing four different scripts in a specific sequence. The dependencies between these scripts are as follows:
- `evaluate_pcrl.py` requires the results from `evaluate_original.py`.
- `evaluate_chatgpt` requires the results from the other experiments

### 1. Evaluate Original Model

Run the `evaluate_original.py` script with the following arguments:
```
python scripts/evaluate_original.py --gen_model=gpt2-xl --bs=16 --results_dir=results
```

### 2. Evaluate Heuristic Model
Run the evaluate_heuristic.py script with the same arguments as the original 

```
python scripts/evaluate_heuristic.py --gen_model=gpt2-xl --bs=16 --results_dir=results
```
### 3. Evaluate PCRL Model
After obtaining the results from evaluate_original.py, run the evaluate_pcrl.py script with the following arguments:
```
python scripts/evaluate_pcrl.py --pcrl_model=gpt2-xl --seed=myseed --gen_model=gpt2-xl --bs=16 --results_dir=results
```

### 4. Compare Results with ChatGPT Metrics
Finally, run the evaluate_chatgpt script with the following arguments:
```
python scripts/evaluate_chatgpt --gen_model=gpt2-xl --eval_model=gpt2-xl --split=seen --seed=myseed
```

## License
The codebase is licensed Apache 2.0 (see LICENSE). The data is a mixture of Self-Instruct (Apache 2.0) and Stanford Alpaca (CC BY-NC 4.0). By training on a mixture of the data, it inherits both licenses.

<!-- Citation
If you use this code or the findings of the study in your research, please cite the corresponding paper: -->
## Acknowledgements

This code references the [RL4LMs by AllenAI](https://github.com/allenai/RL4LMs) and [gisting by Mu et al.](https://github.com/jayelm/gisting). We express our gratitude to the authors for their contributions to the field and for making their code publicly available.

## Contact
For any questions or feedback, please contact nenomigami@gm.gist.ac.kr.
