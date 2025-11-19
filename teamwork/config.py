from dataclasses import dataclass, asdict, fields
import re
from typing import Any

TEAMMATE_FMT = re.compile(r"[a-z]+\.(in|out)")


@dataclass(frozen=True)
class TeamworkConfig:
    """
    Configures the teamwork adaption of a model.

    Teamwork-Specific Attributes:
        teammates: list of all teammates (e.g. 'image.in', 'albedo.out') to support, in the order their parameters are stored
        profile: identifies the default set of layers to adapt and the adapters to use
        lora_rank: the default rank of teamwork LoRAs
        lora_communication: whether to communicate between teammates via LoRA layers, or keep the LoRAs separate (useful as a baseline)
        use_bias: whether to train per-teammate bias for linear layers with a bias

    General Attributes:
        base_model: The checkpoint or model name from which the adapter was or will be trained
        title: A human-readable name for the adapter
        resolution: The resolution the model was finetuned at (eg "1024x1024")
    """

    teammates: list[str]
    lora_rank: int

    base_model: str
    title: str
    resolution: str | None = None

    profile: str | None = None
    lora_communication: bool = True
    use_bias: bool = True

    def __post_init__(self):
        for teammate in self.teammates:
            if TEAMMATE_FMT.fullmatch(teammate) is None:
                raise ValueError(f"{teammate} does not match {TEAMMATE_FMT}")


def config_to_metadata(cfg: TeamworkConfig) -> dict[str, str]:
    """Convert a TeamworkConfig to metadata dictionary format"""
    metadata = {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": f"{cfg.base_model}/teamwork",
        "modelspec.implementation": "samsartor/teamwork",
        "modelspec.title": cfg.title,
        "modelspec.type": "teamwork",
        "base_model": cfg.base_model,
        "teamwork.version": "1.0",
    }
    if cfg.resolution is not None:
        metadata["modelspec.resolution"] = cfg.resolution
    for k, v in asdict(cfg).items():
        if k == "base_model" or k == "title" or k == "resolution":
            pass
        elif k == "teammates":
            metadata["teamwork.teammates"] = ",".join(v)
        elif v is not None:
            metadata[f"teamwork.{k}"] = str(v)
    return metadata


def metadata_to_config(metadata: dict[str, str]) -> TeamworkConfig:
    """Convert metadata dictionary back to TeamworkConfig"""
    cfg_dict: dict[str, Any] = {}
    if metadata.get("modelspec.type") != "teamwork":
        raise ValueError("checkpoint does not appear to be a teamwork model")
    cfg_dict["base_model"] = metadata.get("base_model", "unknown")
    cfg_dict["title"] = metadata.get("modelspec.title", "Teamwork Model")
    cfg_dict["resolution"] = metadata.get("modelspec.resolution")
    known = {f.name for f in fields(TeamworkConfig)}
    for k, v in metadata.items():
        if k.startswith("teamwork."):
            k = k.removeprefix("teamwork.")
            if k not in known:
                continue
            if k == "teammates":
                cfg_dict[k] = v.split(",")
            elif k == "lora_rank":
                cfg_dict[k] = int(v)
            elif k in ["lora_communication", "use_bias"]:
                cfg_dict[k] = v.lower() == "true"
            else:
                cfg_dict[k] = v
    return TeamworkConfig(**cfg_dict)
