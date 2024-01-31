# LLM_ambiguity




## Installation

```bash
conda create -y -n copyright-llm python=3.10 
conda activate copyright-llm

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## How to run

See available model configurations in [`llm_wrapper.py`](./llm_wrapper.py) under `MODEL_CONFIGS` and available runtime parameters in [`run.py`](./run.py) under `RunConfig`.

Example to sequentially run OPT (default 6.7B) and Falcon (default 7B) models and save outputs to custom path:

```bash
python run.py \
    --multirun \
    model_name=dolly,falcon \
    dataset_name=ambig \
    dataset_path=./data/ambig.csv \
    save_path=./hf-outputs
```

Example to run GPT-3.5-Turbo and save results locally to custom path:
```bash
export OPENAI_API_KEY="my-key-123"

python run.py \
    model_name=gpt-3.5 \
    dataset_name=ambig \
    dataset_path=./data/ambig.csv \
    save_strategy=LOCAL \
    save_path=./gpt-outputs
```

The codebase also supports saving the results to WandB by adding `save_strategy=WANDB` as an argument. Before running you also need to export your wandb key or be logged in to wandb.

## How to Cite

```bibtex

```
