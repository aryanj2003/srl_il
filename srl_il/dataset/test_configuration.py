import yaml
from omegaconf import OmegaConf

# Load and print the config
cfg = OmegaConf.load("/home/grimmlins/IL_workspace/srl_il/srl_il/cfg/preset_robomimic/custom_dataset.yaml")
print(OmegaConf.to_yaml(cfg))
