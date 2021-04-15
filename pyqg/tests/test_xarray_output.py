from __future__ import print_function
import numpy as np
import pyqg
import xarray as xr

def test_xarray():
    year = 24*60*60*360.
    m = pyqg.QGModel(tmax=5*year, twrite=10000, tavestart=1*year)
    m.run()
    ds = m.to_dataset()
    
    assert type(ds) == xr.Dataset
    
    expected_vars = ['q', 'u', 'v', 'ph', 'Qy', 'APEflux', 'EKE']  
    for v in expected_vars:
        assert v in ds
    
    
