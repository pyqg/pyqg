import numpy as np
try:
    import xarray as xr
except ImportError:
    raise ImportError(
        "Xarray output in Pyqg requires the Xarray package, which is not installed on your system. " 
        "Please install Xarray in order to activate this feature. "
        "Instructions at http://xarray.pydata.org/en/stable/getting-started-guide/installing.html#instructions"
    )
    
from pyqg.errors import DiagnosticNotFilledError

# Define dict for variable dimensions
spatial_dims = ('time','lev','y','x')
spectral_dims = ('time','lev','l','k')
dim_database = {
    'q': spatial_dims,
    'u': spatial_dims,
    'v': spatial_dims,
    'ufull': spatial_dims,
    'vfull': spatial_dims, 
    'qh': spectral_dims,
    'uh': spectral_dims,
    'vh': spectral_dims,
    'ph': spectral_dims, 
    'dqhdt': spectral_dims, 
    'Ubg': ('lev'),
    'Qy': ('lev'),
}

# dict for variable dimensions
var_attr_database = {
    'q':     { 'units': 's^-1',      'long_name': 'potential vorticity in real space',},
    'u':     { 'units': 'm s^-1',    'long_name': 'zonal velocity anomaly',},
    'v':     { 'units': 'm s^-1',    'long_name': 'meridional velocity anomaly',},
    'ufull': { 'units': 'm s^-1',    'long_name': 'zonal full velocities in real space',},
    'vfull': { 'units': 'm s^-1',    'long_name': 'meridional full velocities in real space',},
    'qh':    { 'units': 's^-1',      'long_name': 'potential vorticity in spectral space',},
    'uh':    { 'units': 'm s^-1',    'long_name': 'zonal velocity anomaly in spectral space',},
    'vh':    { 'units': 'm s^-1',    'long_name': 'meridional velocity anomaly in spectral space',},
    'ph':    { 'units': 'm^2 s^-1',  'long_name': 'streamfunction in spectral space',},
    'p':     { 'units': 'm^2 s^-1',  'long_name': 'streamfunction in real space',},
    'Ubg':   { 'units': 'm s^-1',    'long_name': 'background zonal velocity',},
    'Qy':    { 'units': 'm^-1 s^-1', 'long_name': 'background potential vorticity gradient',} , 
    'dqhdt': { 'units': 's^-2',      'long_name': 'previous partial derivative of potential vorticity wrt. time in spectral space',} , 
    'dqdt':  { 'units': 's^-2',      'long_name': 'previous partial derivative of potential vorticity wrt. time in real space',} , 
}

# dict for variables to convert back from spectral to spatial
vars_to_invert = {
    'dqhdt': 'dqdt',
    'ph': 'p'
}

# dict for coordinate dimensions
coord_database = {
    'time': ('time'),
    'lev': ('lev'),
    'lev_mid': ('lev_mid'),
    'x': ('x'),
    'y': ('y'),
    'l': ('l'),
    'k': ('k'),
}

# dict for coordinate attributes 
coord_attr_database = {
    'time': {'long_name': 'model time', 'units': 's',},
    'lev': {'long_name': 'vertical levels',},
    'lev_mid': {'long_name': 'vertical level interface',},
    'x': {'long_name': 'real space grid points in the x direction', 'units': 'grid point',},
    'y': {'long_name': 'real space grid points in the y direction', 'units': 'grid point',},
    'l': {'long_name': 'spectal space grid points in the l direction', 'units': 'meridional wavenumber',},
    'k': {'long_name': 'spectal space grid points in the k direction', 'units': 'zonal wavenumber',},
}

# list for dataset attributes
attribute_database = [
    'beta',
    'delta',
    'del2',
    'dt',
    'filterfac',
    'L',
    'M',
    'nk',
    'nl',
    'ntd',
    'nx',
    'ny',
    'nz',
    'pmodes',
    'radii',
    'rd',
    'rho',
    'rek',
    'taveint',
    'tavestart',
    'tc',
    'tmax',
    'tsnapint',
    'tsnapstart',
    'twrite',
    'W',
]

# Transform certain key coordinates
transformations = {
    'time': lambda x: np.array([x.t]),
    'lev': lambda x: np.arange(1,x.nz+1),
    'lev_mid': lambda x: np.arange(1.5,x.nz+.5),
    'x': lambda x: x.x[0,:],
    'y': lambda x: x.y[:,0],
    'l': lambda x: x.ll,
    'k': lambda x: x.kk,
}

def model_to_dataset(m):
    '''Convert outputs from model to an xarray dataset'''

    # Create a dictionary of variables
    variables = {}
    for vname in dim_database:
        if hasattr(m,vname):
            data = getattr(m,vname, None).copy()
            if 'time' in dim_database[vname]:
                variables[vname] = (dim_database[vname], data[np.newaxis,...], var_attr_database[vname])
            else:
                variables[vname] = (dim_database[vname], data, var_attr_database[vname])

    # Convert a subset of spectral variables to spatial
    for spectral_var, spatial_var in vars_to_invert.items():
        if hasattr(m, spectral_var):
            data = m.ifft(getattr(m, spectral_var, None))
            variables[spatial_var] = (spatial_dims, data[np.newaxis,...], var_attr_database[spatial_var])

    # Create a dictionary of coordinates
    coordinates = {}
    for cname in coord_database:
        data = transformations[cname](m).copy()
        coordinates[cname] = (coord_database[cname], data, coord_attr_database[cname])

    # Create a dictionary of global attributes
    global_attrs = {}
    for aname in attribute_database:
        if hasattr(m, aname):
            data = getattr(m, aname)
            global_attrs[f"pyqg:{aname}"] = (data)
        
    diagnostics = {}
    for diag_name in m.diagnostics:
        try:
            dims = m.diagnostics[diag_name]['dims']
            data = m.get_diagnostic(diag_name)
            if isinstance(data, np.float64):
                data = np.array([m.get_diagnostic(diag_name)])
            attrs = {'long_name': m.diagnostics[diag_name]['description'], 'units': m.diagnostics[diag_name]['units'],}
            # Ensure time is added to diagnostic data so simulation timesteps
            # can be stacked
            if 'time' not in dims:
                aug_dims = tuple(['time'] + list(dims))
                diagnostics[diag_name] = (aug_dims, data[np.newaxis,...], attrs)
            else:
                diagnostics[diag_name] = (dims, data, attrs)
        except DiagnosticNotFilledError:
            pass
    
    variables.update(diagnostics)
    
    ds = xr.Dataset(variables, coords=coordinates, attrs=global_attrs)
    ds.attrs['title'] = 'pyqg: Python Quasigeostrophic Model'
    ds.attrs['reference'] = 'https://pyqg.readthedocs.io/en/latest/index.html'
    
    return ds
