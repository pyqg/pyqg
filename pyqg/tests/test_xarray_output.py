from __future__ import print_function
import numpy as np
import pyqg
import xarray as xr

def test_xarray():
    # Set parameters & initialize model
    year = 24*60*60*360.
    twrite = 1000
    tavestart = 2*year

    # Initialize QG Model
    m = pyqg.QGModel(tmax=5*year, twrite=twrite, tavestart=tavestart)
    m.run()

    ds = m.to_dataset()
    
    assert type(ds) == xr.Dataset
    
    expected_vars = ['q', 'u', 'v', 'ph', 'Qy']  
    for v in expected_vars:
        assert v in ds
    
    expected_attrs = ['domain length in x direction', 'domain length in y direction', 'numerical timestep', 'title', 'reference'] 
    for a in expected_attrs:
        assert a in ds.attrs

    expected_coords = ['x', 'y', 'l', 'k']  
    for c in expected_coords:
        assert c in ds.coords