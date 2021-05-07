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

# Define dict for diagnostic dimensions
diagnostic_database = {
    'APEflux': ('time','l','k'),
    'APEgen': ('time'),
    'APEgenspec': ('time','l','k'),
    'EKE': ('time','lev'),
    'EKEdiss': ('time'),
    'Ensspec': spectral_dims,
    'KEflux': ('time','l','k'),
    'KEspec': spectral_dims,
    'entspec': ('time','l','k'), 
}

# dict for diagnostics attributes
diagnostic_attr_database = {
    'APEflux': {'long_name': 'spectral flux of available potential energy', 'units': '',},
    'APEgen': {'long_name': 'total APE generation', 'units': '',},
    'APEgenspec': {'long_name': 'spectrum of APE generation', 'units': '',},
    'EKE': {'long_name': 'mean eddy kinetic energy', 'units': '',},
    'EKEdiss': {'long_name': 'total energy dissipation by bottom drag', 'units': '',},
    'Ensspec': {'long_name': 'enstrophy spectrum', 'units': '',},
    'KEflux': {'long_name': 'spectral flux of kinetic energy', 'units': '',},
    'KEspec': {'long_name': 'kinetic energy spectrum', 'units': '',},
    'entspec': {'long_name': 'barotropic enstrophy spectrum', 'units': '',},
}    

# dict for coordinate dimensions
coord_database = {
    'time': ('time'),
    'lev': ('lev'),
    'x': ('x'),
    'y': ('y'),
    'l': ('l'),
    'k': ('k'),
}

# dict for coordinate attributes 
coord_attr_database = {
    'time': {'long_name': 'model time', 'units': 'seconds',},
    'lev': {'long_name': 'vertical levels',},
    'x': {'long_name': 'real space grid points in the x direction', 'units': 'grid point',},
    'y': {'long_name': 'real space grid points in the y direction', 'units': 'grid point',},
    'l': {'long_name': 'spectal space grid points in the l direction', 'units': 'meridional wavenumber',},
    'k': {'long_name': 'spectal space grid points in the k direction', 'units': 'zonal wavenumber',},
}

# dict for global attributes
attribute_database = {
    'L': (),
    'W': (),
    'dt': (),
    'filterfac': (),
    'nk': (),
    'nl': (),
    'ntd': (),
    'nx': (),
    'ny': (),
    'nz': (),
    'pmodes': (),
    'radii': (),
    'rek': (),
    'taveint': (),
    'tavestart': (),
    'tc': (),
    'time': (),
    'tmax': (),
    'tsnapint': (),
    'tsnapstart': (),
    'twrite': (),
}

# Transform certain key coordinates
transformations = {
    'time': lambda x: [x.t],
    'lev': lambda x: np.arange(1,x.nz+1),
    'x': lambda x: x.x[0,:],
    'y': lambda x: x.y[:,0],
    'l': lambda x: x.ll,
    'k': lambda x: x.kk,
}

def model_to_dataset(m):
    '''Convert output from model to an xarray dataset'''

    diagnostics = {}
    for dname in diagnostic_database:
        try: 
            data = m.get_diagnostic(dname)
            if 'time' in diagnostic_database[dname]:
                diagnostics[dname] = (diagnostic_database[dname], data[np.newaxis,...], diagnostic_attr_database[dname])
            else:
                diagnostics[dname] = (diagnostic_database[dname], data, diagnostic_attr_database[dname])   
        except:
            diagnostics[dname] = np.nan
            
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
        if cname in transformations:
            data = transformations[cname](m)
        else:
            if hasattr(m, cname):
                data = getattr(m, cname)
        coordinates[cname] = (coord_database[cname], data, coord_attr_database[cname])
    
    # Create a dictionary of global attributes
    global_attrs = {}
    for aname in attribute_database:
        if hasattr(m, aname):
            data = getattr(m, aname)
            global_attrs[aname] = (data)
        
    variables.update(diagnostics)
    ds = xr.Dataset(variables, coords=coordinates, attrs=global_attrs)
    ds.attrs['title'] = 'pyqg: Python Quasigeostrophic Model'
    ds.attrs['references'] = 'https://pyqg.readthedocs.io/en/latest/index.html'
   
    return ds 
