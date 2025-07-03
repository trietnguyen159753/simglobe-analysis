from scipy.stats._stats_py import trim_mean
import tomli
from pydantic import BaseModel


class PipelineConfig(BaseModel):
    project_name: str
    input_var: list[str]
    output_var: list[str]
    scenario: list[str]
    unique: list[str]
    trim_radius: float


def load_config(config_path: str = "config.toml") -> PipelineConfig:
    with open(config_path, "rb") as f:
        config_data = tomli.load(f)
    return PipelineConfig(**config_data)


config = load_config()
