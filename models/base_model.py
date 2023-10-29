import os
import dataclasses
import hashlib
import json
from typing import Dict, Any
from torch import nn
import torch
import copy

def flat_dict(data, prefix: str = ""):
    out_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out_data.update(flat_dict(v, prefix=f"{prefix}{k}."))
        else:
            if v is not None:
                out_data[prefix+k] = v
    return out_data

class BaseModel(nn.Module):
    name: str = None

    def __init__(self):
        super().__init__()

    def as_dict(self, flatten: bool = False) -> Dict[str, Any]:
        data = {
            "class": type(self).name,
            "dataset_name": self.dataset_name,
            "base": dataclasses.asdict(self.base_params),
            "share": dataclasses.asdict(self.share_params),
            "train": copy.deepcopy(self.train_params),
        }

        if data["train"]["model_id"] == 0:
            del(data["train"]["model_id"])

        if flatten:
            data = flat_dict(data)

        return data

    def md5(self) -> str:
        return hashlib.md5(
            json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def __hash__(self):
        return hash((type(self), dataclasses.astuple(self.base_params)))

    @staticmethod
    def is_legal(
        base_config: Dict[str, Any], share_config: Dict[str, Any]
    ) -> bool:
        raise NotImplementedError()

    def get_model_path(self) -> str:
        return f"results/{self.name}/{self.md5()}/model.pkl"

    def exists(self):
        os.path.isfile(self.get_model_path())

    def save(self):
        path = self.get_model_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self):
        self.load_state_dict(torch.load(self.get_model_path()))

    def get_base_model(self) -> "BaseModel":
        return type(self)(
            base_params=dataclasses.asdict(self.base_params),
            share_params={},
            train_params=self.train_params,
            dataset_name=self.dataset_name,
        )

    def get_params(self):
        total_params = sum(param.numel() for param in self.parameters())
        return total_params
