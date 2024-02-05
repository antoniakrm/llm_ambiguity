from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import docx
import pandas as pd
from omegaconf import DictConfig


@dataclass
class InputExample:
    sentence: str
    change_type: str
    ambig_status: int


class Dataset(ABC):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def _load_dataset(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def get_prompts(self, *args, **kwargs) -> List[str]: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]: ...

    def __len__(self) -> int:
        return len(self.data)


class AmbiguityDataset(Dataset):
    def __init__(self, csv_filepath: Path):
        self.name = "ambiguous_sentences"

        self._load_dataset(csv_filepath)

    def _load_dataset(self, csv_filepath: Path) -> None:
        csv = pd.read_csv(csv_filepath, sep=';')
        self.data: List[InputExample] = [
            InputExample(sentence, change_type, ambig_status)
            for sentence, change_type, ambig_status in zip(csv["sentence"], csv["change_type"], csv["is_ambiguous"])
        ]

    def get_prompts(self, amb_sent: InputExample) -> List[str]:
        prompt_b = (
                f"Is the following sentence a pun? Reply only with yes or no. The sentence is: {amb_sent.sentence}\n\n"
        )
        return [prompt_b]

    def __getitem__(self, index: int) -> Dict[str, Union[InputExample, List[str]]]:
        amb_sent = self.data[index]
        prompts = self.get_prompts(amb_sent)

        return {"data": amb_sent, "prompts": prompts}


def get_dataset(config: DictConfig) -> Dataset:
    if config.dataset_name == "ambig":
        return AmbiguityDataset(Path(config.dataset_path))
    else:
        raise ValueError(f"Currenlty only the ambiguity dataset is supported. Invalid dataset name: {config.dataset_name}")
