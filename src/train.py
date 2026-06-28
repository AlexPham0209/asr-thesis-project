import hydra
from omegaconf import OmegaConf
import torch
from transformers import Trainer, TrainingArguments
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    print("------- Running Experiment Configuration -------")
    print(OmegaConf.to_yaml(cfg))

    if not cfg.get("model"):
        raise ValueError("Missing 'model' configuration block in your YAML")

    # Instantiating model
    print("------- Model Configurations -------")
    print(f"{cfg.model}\n")

    print("------- Instantiating Model from Configuration -------")
    model = instantiate(cfg.model)

    # Creating Dataset and Dataloader
    print("\n------- Instantiating Dataset and Dataloader from Configuration -------")
    print(f"{cfg.dataset}\n")
    dataset = instantiate(cfg.dataset)


if __name__ == "__main__":
    main()
