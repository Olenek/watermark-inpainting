from dataclasses import dataclass


@dataclass
class SLBRConfig:
    sim_metric: str = 'cos'
    k_center: int = 1
    mask_mode: str = 'cat'
    bg_mode: str = 'res_mask'


cfg = SLBRConfig()