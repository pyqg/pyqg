from __future__ import print_function
import numpy as np
import pyqg
import xarray as xr

def test_xarray():
    m = pyqg.QGModel(1)
    ds = m.to_dataset()
    
    assert type(ds) == xr.Dataset
    
    expected_vars = ['q', 'u', 'v', 'ph', 'Qy']  
    for v in expected_vars:
        assert v in ds
    
    
