from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path

from src.core.reranker import RerankerConfig
from src.core.retriever import RankFusionConfig
from src.retrievers.global_retriever import GlobalRetrieverConfig
from src.retrievers.local_retriever import LocalRetrieverConfig


class SystemConfig(BaseModel):
    model_cache_dir: str
    output_dir: str
    log_level: str


class DataConfig(BaseModel):
    law_path: str
    wseg_law_path: str
    queries_path: str
    wseg_queries_path: str
    output_path: str


class VectorDBConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTOR_DB_")

    collection_name: str
    host: str = "localhost"
    port: int = 6333
    api_key: str | None = None  # Optional API key


class PipelineConfig(BaseModel):
    enable_local_retriever: bool
    enable_global_retriever: bool
    enable_reranker: bool
    save_local_path: str | None = None
    save_global_path: str | None = None
    save_fused_path: str | None = None
    save_reranked_path: str | None = None
    save_results_path: str | None = None
    local_retriever: LocalRetrieverConfig
    global_retriever: GlobalRetrieverConfig
    rank_fusion: RankFusionConfig
    reranker: RerankerConfig


class Config(BaseModel):
    system: SystemConfig
    data: DataConfig
    pipeline: PipelineConfig


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """Loads configuration from a YAML file and validates it with Pydantic."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)
