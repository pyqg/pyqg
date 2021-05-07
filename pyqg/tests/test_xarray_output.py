from __future__ import print_function
import numpy as np
import pyqg
import xarray as xr

def test_xarray():
    m = pyqg.QGModel(1)
    ds = model_to_dataset(m) 
    
    assert type(ds) == xr.Dataset
    
    expected_vars = ['q', 'u', 'v', 'ph', 'Qy']  
    for v in expected_vars:
        assert v in ds
        
    expected_attrs = ['L', 'W', 'dt', 'title', 'references']  
    for a in expected_attrs:
        assert a in ds.attrs
        
    expected_coords = ['time', 'x', 'y', 'l', 'k']  
    for c in expected_coords:
        assert c in ds.coords
    
    
