"""
Configuration management for SIM-APA v2.0.
"""

from pathlib import Path

from omegaconf import OmegaConf, DictConfig


class Config:
    """
    Loads and manages all project YAML configuration files.
    """

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self.cfg = self._load_configs()

    def _load_configs(self) -> DictConfig:
        config = OmegaConf.create()
        yaml_files = sorted(self.config_dir.glob("*.yaml"))

        if not yaml_files:
            raise FileNotFoundError(
                f"No configuration files found in {self.config_dir}"
            )

        for yaml_file in yaml_files:
            current_cfg = OmegaConf.load(yaml_file)
            config = OmegaConf.merge(config, current_cfg)

        return config

    def __getattr__(self, item):
        return getattr(self.cfg, item)

    def __repr__(self) -> str:
        return OmegaConf.to_yaml(self.cfg)
