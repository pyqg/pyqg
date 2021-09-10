from __future__ import print_function
import numpy as np
import pyqg
import xarray as xr
import pytest

year = 24*60*60*360.

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

@pytest.fixture
def QGModel():
    '''Two-layer Model'''
    m = pyqg.QGModel(tmax=year/2, twrite=10000, tavestart=year/3)
    return m

@pytest.fixture
def LayeredModel():
    '''Layered quasigeostrophic model'''
    m = pyqg.LayeredModel(tmax=year/2, twrite=10000, tavestart=year/3)
    return m

@pytest.fixture
def BTModel():
    '''Barotropic model'''
    m = pyqg.BTModel(L=2.*np.pi, nx=256, beta=0., H=1., rek=0., 
                     rd=None, tmax=40, dt=0.001, taveint=1, ntd=4)
    return m
    
@pytest.fixture
def SQGModel():
    '''Surface Quasi-Geostrophic Model'''
    m = sqg_model.SQGModel(L=2.*pi, nx=512, tmax = 26.005, beta = 0., 
                           Nb = 1., H = 1., rek = 0., rd = None, dt = 0.005,
                           taveint=1, twrite=400, ntd=4)
    return m
    
@pytest.mark.parametrize("model", [QGModel(), LayeredModel(), BTModel(), SQGModel()])
def test_xarray(model):
    '''Run with snapshots and test contents of xarray.dataset'''
    m = model()
    for snapshot in m.run_with_snapshots(tsnapstart=m.t, tsnapint=year/4):
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