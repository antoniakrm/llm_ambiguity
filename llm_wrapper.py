import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import openai
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM, LlamaTokenizerFast)

from dataset import Dataset, InputExample


class SaveStrategy(Enum):
    LOCAL = "local"
    WANDB = "wandb"


@dataclass
class Completion:
    prompt: str
    response: str


@dataclass
class ModelOutput:
    model_id: str
    sentence: str
    change_type: str
    ambig_status: bool
    completions: List[Completion]

    def as_nested_list(self) -> List[List[str]]:
        return [
            [self.model_id, self.sentence, self.change_type, self.ambig_status, c.prompt, c.response]
            for c in self.completions
        ]


class Wrapper(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        self.model_config = MODEL_CONFIGS[config.model_name]
        self.resolved_model_id = self.model_config.model_id

    @abstractmethod
    def inference_step(
        self, example: Dict[str, Union[InputExample, List[str]]]
    ) -> ModelOutput: ...

    def save_outputs(self, outputs: List[ModelOutput], dataset_name: str) -> pd.DataFrame:
        # Flatten
        all_outputs = []
        for output in outputs:
            all_outputs.extend(output.as_nested_list())

        # Convert to dataframe
        df = pd.DataFrame(
            all_outputs, columns=["model_id", "sentence", "content", "ambig_status", "prompt", "response"]
        )

        # Save
        if self.config.save_strategy == SaveStrategy.LOCAL:
            Path.mkdir(Path(self.config.save_path), parents=True, exist_ok=True)
            output_filepath = (
                Path(self.config.save_path)
                / f"{os.path.basename(self.resolved_model_id)}_{dataset_name}_outputs.csv"
            )
            df.to_csv(output_filepath, sep="\t", index=False)

            print(f"Saved outputs to `{output_filepath}`")

        elif self.config.save_strategy == SaveStrategy.WANDB:
            wandb.init(
                project=dataset_name,
                name=os.path.basename(self.resolved_model_id),
                id=f"{os.path.basename(self.resolved_model_id)}_{dataset_name}",
                config=OmegaConf.to_container(self.config),
            )
            wandb.log({"Result": wandb.Table(dataframe=df)})
            wandb.finish()

            print(f"Saved outputs to wandb")

        return df

    def run_inference(self, ds: Dataset) -> pd.DataFrame:
        model_outputs = [self.inference_step(ds[i]) for i in tqdm(range(len(ds)))]
        return self.save_outputs(model_outputs, dataset_name=ds.name)


class OpenaiWrapper(Wrapper):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        print("Using OpenAI API")

        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def inference_step(self, example: Dict[str, Union[InputExample, List[str]]]) -> ModelOutput:
        completions = []

        for prompt_text in example["prompts"]:
            while True:
                try:
                    response = self.client.chat.completions.create(
                        model=self.resolved_model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt_text,
                            }
                        ],
                        **self.model_config.generation_kwargs,
                    )
                    break
                except openai.OpenAIError as e:
                    #if e.status == 429:
                     #   print("Rate limit error, waiting 20 seconds before trying again")
                      #  time.sleep(20)
                    #else:
                        print(f"Error: {e}")
                        raise e

            generated_text = response.choices[0].message.content
            completions.append(Completion(prompt_text, generated_text))

        model_output = ModelOutput(
            model_id=self.resolved_model_id,
            sentence=example["data"].sentence,
            change_type=example["data"].change_type,
            ambig_status=example["data"].ambig_status,
            completions=completions,
        )
        return model_output

class HfWrapper(Wrapper):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.model_size = self.config.model_size or self.model_config.default_model_size
        self.resolved_model_id = self.model_config.model_id.format(model_size=self.model_size)

        print(f"Using HuggingFace with local model: {self.resolved_model_id}")

        self.model = self.model_config.model_cls(
            pretrained_model_name_or_path=self.resolved_model_id,
            torch_dtype=self.model_config.dtype,
        )
        self.tokenizer = self.model_config.tokenizer_cls(
            pretrained_model_name_or_path=self.resolved_model_id
        )

    def inference_step(self, example: Dict[str, Union[InputExample, List[str]]]) -> ModelOutput:
        if self.model_config.uses_chat_template:
            print("Using chat template")
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], return_tensors="pt"
                ).to(self.model.device)
                for prompt in example["prompts"]
            ]
        else:
            prompts = [
                self.tokenizer(p, return_tensors="pt").input_ids.to(self.model.device)
                for p in example["prompts"]
            ]

        completions = []
        for prompt in prompts:
            # Generate
            generated_tokens = self.model.generate(
                prompt,
                **self.model_config.generation_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generated_tokens = generated_tokens[:, prompt.shape[-1] :].cpu()

            # Decode
            prompt_text = self.tokenizer.batch_decode(prompt, skip_special_tokens=True)[0]
            generated_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]

            completions.append(Completion(prompt_text, generated_text))

        model_output = ModelOutput(
            model_id=self.resolved_model_id,
            sentence=example["data"].sentence,
            change_type=example["data"].change_type,
            ambig_status=example["data"].ambig_status,
            completions=completions,
        )
        return model_output


class ApiType(Enum):
    OPENAI = "openai"
    HF = "hf"


@dataclass
class OpenaiApiConfig:
    model_id: str = "gpt-3.5-turbo"
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"temperature": 1.0, "max_tokens": 200}
    )
    api_type: ApiType = ApiType.OPENAI
    wrapper_cls: Wrapper = OpenaiWrapper

@dataclass
class HFApiConfig:
    model_id: str
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 20,
            "num_return_sequences": 1,
        }
    )
    dtype: torch.dtype = torch.float16
    default_model_size: str = "7b"
    model_cls: Callable = partial(AutoModelForCausalLM.from_pretrained, device_map="auto")
    tokenizer_cls: Callable = partial(AutoTokenizer.from_pretrained, use_fast=True)
    uses_chat_template: bool = False
    api_type: ApiType = ApiType.HF
    wrapper_cls: Wrapper = HfWrapper


MODEL_CONFIGS = {
    "gpt-3.5": OpenaiApiConfig(),
    "gpt-4": OpenaiApiConfig(model_id="gpt-4"),
    "llama": HFApiConfig(
        model_id="meta-llama/Llama-2-{model_size}-hf",
        dtype=torch.bfloat16,
        tokenizer_cls=LlamaTokenizerFast.from_pretrained,
    ),
    
    "dolly": HFApiConfig(model_id="databricks/dolly-v2-{model_size}", dtype=torch.bfloat16),
    
    "falcon": HFApiConfig(model_id="tiiuae/falcon-{model_size}", dtype=torch.bfloat16),
    "falcon-instruct": HFApiConfig(
        model_id="tiiuae/falcon-{model_size}-instruct", dtype=torch.bfloat16
    ),
    "opt": HFApiConfig(
        model_id="facebook/opt-{model_size}",
        tokenizer_cls=partial(AutoTokenizer.from_pretrained, use_fast=False),
        default_model_size="6.7b",
    ),
    "pythia": HFApiConfig(
        model_id="EleutherAI/pythia-{model_size}",
        model_cls=partial(
            GPTNeoXForCausalLM.from_pretrained,
            device_map="auto",
        ),
        default_model_size="6.9b",
    ),
    "mistral": HFApiConfig(
        model_id="mistralai/Mistral-{model_size}-v0.1",
        default_model_size="7B",
    ),
    "mistral-instruct": HFApiConfig(
        model_id="mistralai/Mistral-{model_size}-Instruct-v0.2",
        default_model_size="7B",
        uses_chat_template=True,
    ),
}


def get_llm_wrapper(config: DictConfig) -> Wrapper:
    if config.model_name not in MODEL_CONFIGS:
        raise RuntimeError(f"Model {config.model_name} is not supported.")

    return MODEL_CONFIGS[config.model_name].wrapper_cls(config)
