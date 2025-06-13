import tomli
from pydantic import BaseModel


class PathSettings(BaseModel):
    input_constant: str
    input_random: str
    output_summary: str
    output_cook: str
    output_dist: str


class FilterParams(BaseModel):
    iqr_threshold: int
    filter_enable: bool


class PipelineConfig(BaseModel):
    project_name: str
    path: PathSettings
    input_var: list[str]
    output_var: list[str]
    scenario: list[str]
    unique: list[str]
    filter_param: FilterParams


def load_config(config_path: str = "config.toml") -> PipelineConfig:
    with open(config_path, "rb") as f:
        config_data = tomli.load(f)
    return PipelineConfig(**config_data)


config = load_config()
