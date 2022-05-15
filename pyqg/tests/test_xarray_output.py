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
    'p',
    'dqdt',
    'qh', 
    'uh', 
    'vh', 
    'ph', 
    'Ubg', 
    'Qy',
    'dqhdt',
]

expected_diags = [
    'EKE', 
    'entspec', 
    'APEflux', 
    'APEflux_div', 
    'KEflux', 
    'KEflux_div', 
    'APEgen',
    'APEgenspec',
    'Ensspec',
    'KEspec',
    'EKEdiss',
    'KEspec_modal',
    'PEspec_modal',
    'APEspec',
    'ENSflux',
    'ENSgenspec',
    'paramspec',
    'paramspec_APEflux',
    'paramspec_KEflux',
    'Dissspec',
    'KEfrictionspec',
    'ENSDissspec',
    'ENSfrictionspec',
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

def QG():
    '''Initialize Two-layer Model'''
    return pyqg.QGModel(tmax=year/2, twrite=1000, tavestart=year/3)

def Layered():
    '''Initialize Layered quasigeostrophic model'''
    L =  1000.e3     # length scale of box    [m]
    Ld = 15.e3       # deformation scale      [m]
    kd = 1./Ld       # deformation wavenumber [m^-1]
    Nx = 64          # number of grid points

    H1 = 500.        # layer 1 thickness  [m]
    H2 = 1750.       # layer 2
    H3 = 1750.       # layer 3

    U1 = 0.05          # layer 1 zonal velocity [m/s]
    U2 = 0.025         # layer 2
    U3 = 0.00          # layer 3

    rho1 = 1025.
    rho2 = 1025.275
    rho3 = 1025.640

    rek = 1.e-7       # linear bottom drag coeff.  [s^-1]
    f0  = 0.0001236812857687059 # coriolis param [s^-1]
    beta = 1.2130692965249345e-11 # planetary vorticity gradient [m^-1 s^-1]

    return pyqg.LayeredModel(nx=Nx, nz=3, U=[U1,U2,U3], V=[0.,0.,0.], L=L, f=f0, beta=beta,
                             H=[H1,H2,H3], rho=[rho1,rho2,rho3], rek=rek,
                             tmax=year/2, twrite=1000, tavestart=year/3)

def BT():
    '''Initialize Barotropic model'''
    return pyqg.BTModel(L=2.*np.pi, nx=256, beta=0., H=1., rek=0., 
                        rd=None, tmax=year/2, twrite=1000, tavestart=year/3,
                        dt=year/8, taveint=1, ntd=4)
    
def SQG():
    '''Initialize Surface Quasi-Geostrophic Model'''
    return pyqg.SQGModel(L=2.*np.pi, nx=512, tmax=year/2, beta=0.,
                         Nb=1., H=1., f_0 = 1., dt=year/8,
                         taveint=1, twrite=1000, ntd=4, tavestart=year/3)
    
@pytest.fixture(params=[QG, Layered, BT, SQG])
def all_models(request):
    model = request.param()
    return model

def test_xarray(all_models):
    '''Run with snapshots and test contents of xarray.dataset'''
    m=all_models
    tsnapint=year/4
    datasets = []
    timevals = []

    for snapshot in m.run_with_snapshots(tsnapstart=m.t, tsnapint=tsnapint):
        ds = m.to_dataset()
        assert type(ds) == xr.Dataset

        datasets.append(ds)
        timevals.append(m.t)

        if snapshot < tsnapint:

            for v in expected_vars:
                assert v in ds

            for a in expected_attrs:
                assert f"pyqg:{a}" in ds.attrs

            for c in expected_coords:
                assert c in ds.coords

        if snapshot > tsnapint:

            for v in list(ds.keys()): 
                assert v in expected_vars + expected_diags

            for v in expected_diags:
                if v in ds:
                    assert 'time' in ds[v].coords

            for a in expected_attrs:
                assert f"pyqg:{a}" in ds.attrs

            for c in expected_coords:
                assert c in ds.coords

    concatenated = xr.concat(datasets, dim='time')

    assert np.allclose(concatenated.coords['time'], timevals)
