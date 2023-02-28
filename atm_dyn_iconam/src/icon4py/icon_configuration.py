from dataclasses import dataclass

from icon4py.diffusion.diffusion import DiffusionConfig

@dataclass
class IconRunConfig:
    n_time_steps: int
    dtime: float
@dataclass
class AtmoNonHydroConfig:
    n_substeps: int = 5
@dataclass
class IconConfig:
    run_config: IconRunConfig
    diffusion_config:DiffusionConfig
    dycore_config: AtmoNonHydroConfig

def read_config() -> IconConfig:
    model_run_config = IconRunConfig(n_time_steps=5, dtime=10.0)
    diffusion_config : DiffusionConfig = DiffusionConfig()
    dycore_config = AtmoNonHydroConfig(n_substeps=5)
    return IconConfig(run_config=model_run_config, diffusion_config=diffusion_config, dycore_config=dycore_config)

