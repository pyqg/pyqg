from __future__ import print_function
import numpy as np
import pyqg
import xarray as xr

def test_xarray():
    m = pyqg.QGModel(1)
    ds = m.to_dataset()
    
    assert type(ds) == xr.Dataset

if __name__ == "__main__":
    test_xarray()