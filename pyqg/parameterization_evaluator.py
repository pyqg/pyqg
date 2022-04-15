import numpy as np
import xarray as xr
from functools import cached_property

def _ensure_xarray(model):
    if isinstance(model, xr.Dataset):
        return model
    else:
        return model.to_dataset()

class ParameterizationEvaluator:
    def __init__(self, target, baseline):
        self.target = _ensure_xarray(target)
        self.baseline = _ensure_xarray(baseline)

    def distances_wrt_target(self, model):
        pass

    @cached_property
    def baseline_distances(self):
        return self.distances_wrt_target(self.baseline)

    def improvement_over_baseline(self, model):
        pass


