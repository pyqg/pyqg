import numpy as np
import xarray as xr

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
    'Ubg': ('lev'),
    'Qy': ('lev'),
}

# dict for variable dimensions
var_attr_database = {
    'q': {'long_name': 'potential vorticity in real space', 'units': 'meters squared Kelvin per second per kilogram',},
    'u': {'long_name': 'zonal velocity anomaly', 'units': 'meters per second',},
    'v': {'long_name': 'meridional velocity anomaly', 'units': 'meters per second',},
    'ufull': {'long_name': 'zonal full velocities in real space', 'units': 'meters per second',},
    'vfull': {'long_name': 'meridional full velocities in real space', 'units': 'meters per second',},
    'qh': {'long_name': 'potential vorticity in spectral space', 'units': 'meters squared Kelvin per second per kilogram',},
    'uh': {'long_name': 'zonal velocity anomaly in spectral space', 'units': 'meters per second',},
    'vh': {'long_name': 'meridional velocity anomaly in spectral space', 'units': 'meters per second',},
    'ph': {'long_name': 'streamfunction in spectral space', 'units': 'meters squared per second',},
    'Ubg': {'long_name': 'background zonal velocity', 'units': 'meters per second',},
    'Qy': {'long_name': 'background potential vorticity gradient', 'units': 'meters squared Kelvin per second per kilogram',} ,
 
}

# dict for coordinate dimensions
coord_database = {
    'time': ('time'),
    'lev': ('lev'),
    'x': ('x'),
    'y': ('y'),
    'l': ('l'),
    'k': ('k'),
    'nx': (),
    'ny': (),
    'nz': (),
    'nl': (),
    'nk': (),
    'rek': (),
    'tc': (),
    'dt': (),
    'L': (),
    'W': (),
    'filterfac': (),
    'twrite': (),
    'tmax': (),
    'tavestart': (),
    'tsnapstart': (),
    'taveint': (),
    'tsnapint': (),
    'ntd': (),
    'pmodes': (),
    'radii': (),
}

# dict for coordinate attributes 
coord_attr_database = {
    'time': {'long_name': 'model time', 'units': 'seconds',},
    'lev': {'long_name': 'vertical levels',},
    'x': {'long_name': 'real space grid points in the x direction', 'units': 'grid point',},
    'y': {'long_name': 'real space grid points in the y direction', 'units': 'grid point',},
    'l': {'long_name': 'spectal space grid points in the l direction', 'units': 'meridional wavenumber',},
    'k': {'long_name': 'spectal space grid points in the k direction', 'units': 'zonal wavenumber',},
    'nx': {'long_name': 'number of real space grid points in x direction',},
    'ny': {'long_name': 'number of real space grid points in y direction (default: nx)'},
    'nz': {'long_name': 'number of vertical levels',},
    'nl': {'long_name': 'number of spectral space grid points in l direction', 'units': 'grid point',},
    'nk': {'long_name': 'number of spectral space grid points in k direction', 'units': 'grid point',},
    'rek': {'long_name': 'linear drag in lower layer', 'units': 'per second',},
    'tc': {'long_name': 'model timestep', 'units': 'seconds',},
    'dt': {'long_name': 'numerical timestep', 'units': 'seconds',},
    'L': {'long_name': 'domain length in x direction', 'units': 'meters',},
    'W': {'long_name': 'domain length in y direction', 'units': 'meters',},
    'filterfac': {'long_name': 'amplitude of spectral spherical filter', 'units': '',},
    'twrite': {'long_name': 'interval for cfl writeout', 'units': 'number of timesteps',},
    'tmax': {'long_name': 'total time of integration', 'units': 'seconds',},
    'tavestart': {'long_name': 'start time for averaging', 'units': 'seconds',},
    'tsnapstart': {'long_name': 'start time for snapshot writeout', 'units': 'seconds'},
    'taveint': {'long_name': 'time interval for accumulation of diagnostic averages', 'units': 'seconds'},
    'tsnapint': {'long_name': 'time interval for snapshots', 'units': 'seconds',},
    'ntd': {'long_name': 'number of threads used',},
    'pmodes': {'long_name': 'vertical pressure modes',},
    'radii': {'long_name': 'deformation radii', 'units': 'meters',},
}

# dict for dataset attributes
global_attrs = {
    'title': 'pyqg: Python Quasigeostrophic Model',
    'references': 'https://pyqg.readthedocs.io/en/latest/index.html',
}

# Transform certain key coordinates
transformations = {
    'time': lambda x: np.array([x.t]),
    'lev': lambda x: np.arange(1,x.nz+1),
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
            data = getattr(m,vname, None)
            if 'time' in dim_database[vname]:
                variables[vname] = (dim_database[vname], data[np.newaxis,...], var_attr_database[vname])
            else:
                variables[vname] = (dim_database[vname], data, var_attr_database[vname])

    # Create a dictionary of coordinates
    coordinates = {}
    for cname in coord_database:
        if hasattr(m, cname):
            if cname in transformations:
                data = transformations[cname](m)
            else:
                data = getattr(m, cname)
            coordinates[cname] = (coord_database[cname], data, coord_attr_database[cname])
        
    ds = xr.Dataset(variables, coords=coordinates, attrs=global_attrs)

    return ds
