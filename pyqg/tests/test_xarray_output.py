from __future__ import print_function
import numpy as np
import pyqg
import xarray as xr

def test_xarray():
    expected_vars = [
        'q', 
        'u', 
        'v', 
        'ufull', 
        'vfull', 
        'qh', 
        'uh', 
        'vh', 
        'ph', 
        'Ubg', 
        'Qy',
    ]

    expected_diags = [
        'EKE', 
        'entspec', 
        'APEflux', 
        'KEflux', 
        'APEgenspec', 
        'APEgen'
    ]

    expected_attrs = [
        'L',
        'W',
        'dt',
        'filterfac',
        'nk',
        'nl',
        'ntd',
        'nx',
        'ny',
        'nz',
        'rek',
        'taveint',
        'tavestart',
        'tc',
        'tmax',
        'twrite',
    ]

    expected_coords = [
        'time',
        'lev',
        'x',
        'y',
        'l',
        'k',
    ]
    
    # Initialize QG Model
    year = 24*60*60*360.
    m = pyqg.QGModel(tmax=year/2, twrite=10000, tavestart=year/3)
    tsnapint = year/4
    
    # Run with snapshots
    for snapshot in m.run_with_snapshots(tsnapstart=m.t, tsnapint=tsnapint):
        ds = m.to_dataset()
        assert type(ds) == xr.Dataset

        if snapshot < tsnapint:

            for v in expected_vars:
                assert v in ds

            for a in expected_attrs:
                assert f"pyqg:{a}" in ds.attrs

            for c in expected_coords:
                assert c in ds.coords

        if snapshot > tsnapint:

            for v in expected_vars + expected_diags:
                assert v in ds

            for a in expected_attrs:
                assert f"pyqg:{a}" in ds.attrs

            for c in expected_coords:
                assert c in ds.coords