import numpy as np
import xarray as xr

def model_to_dataset(m):
    '''Convert outputs from model to an xarray dataset'''

   # Define dict for variable dimensions
    spatial_dims = ('time','z','y','x')
    spectral_dims = ('time','z','l','k')
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
        'Ubg': ('z'),
        'Qy': ('z'),
    }

    # dict for variable dimensions
    var_attr_database = {
        'q': {'long_name': 'potential vorticity in real space', 'units': 'm\u00B2 s\u207B\u00B9 K kg\u207B\u00B9',},
        'u': {'long_name': 'zonal velocity anomaly', 'units': 'm s\u207B\u00B9',},
        'v': {'long_name': 'meridional velocity anomaly', 'units': 'm s\u207B\u00B9',},
        'ufull': {'long_name': 'zonal full velocities in real space', 'units': 'm s\u207B\u00B9',},
        'vfull': {'long_name': 'meridional full velocities in real space', 'units': 'm s\u207B\u00B9',},
        'qh': {'long_name': 'potential vorticity in spectral space', 'units': 'm\u00B2 s\u207B\u00B9 K kg\u207B\u00B9',},
        'uh': {'long_name': 'zonal velocity anomaly in spectral space', 'units': 'm s\u207B\u00B9',},
        'vh': {'long_name': 'meridional velocity anomaly in spectral space', 'units': 'm s\u207B\u00B9',},
        'ph': {'long_name': 'streamfunction in spectral space', 'units': 'm\u00B2 s\u207B\u00B9',},
        'Ubg': {'long_name': 'background zonal velocity', 'units': 'm s\u207B\u00B9',},
        'Qy': {'long_name': 'background potential vorticity gradient', 'units': 'm\u00B2 s\u207B\u00B9 K kg\u207B\u00B9',} ,
        'kk': {'long_name': 'zonal wavenumbers', 'units': 'm\u207B\u00B9',} ,
        'll': {'long_name': 'meridional wavenumbers', 'units': 'm\u207B\u00B9',} ,
    }

    # dict for coordinate dimensions
    coord_database = {
        'time': ('time'),
        'z': ('z'),
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
        'z': {'long_name': 'vertical levels', 'units': 'none',},
        'x': {'long_name': 'real space grid points in the x direction', 'units': 'grid point',},
        'y': {'long_name': 'real space grid points in the y direction', 'units': 'grid point',},
        'l': {'long_name': 'spectal space grid points in the l direction', 'units': 'meridional wavenumber',},
        'k': {'long_name': 'spectal space grid points in the k direction', 'units': 'zonal wavenumber',},
        'nx': {'long_name': 'number of real space grid points in x direction', 'units': 'none',},
        'ny': {'long_name': 'number of real space grid points in y direction (default: nx)', 'units': 'none',},
        'nz': {'long_name': 'number of vertical levels', 'units': 'none',},
        'nl': {'long_name': 'number of spectral space grid points in l direction', 'units': 'grid point',},
        'nk': {'long_name': 'number of spectral space grid points in k direction', 'units': 'grid point',},
        'rek': {'long_name': 'linear drag in lower layer', 'units': 'seconds\u207B\u00B9',},
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
        'ntd': {'long_name': 'number of threads used', 'units': 'none',},
        'pmodes': {'long_name': 'vertical pressure modes', 'units': 'none',},
        'radii': {'long_name': 'deformation radii', 'units': 'meters',},
    }

    # dict for dataset attributes
    ds_attr_database = {
        'title': 'pyqg: Python Quasigeostrophic Model',
        'institution': '',
        'source': ('version: {}'.format(pyqg.__version__)),
        'history': '',
        'references': 'https://pyqg.readthedocs.io/en/latest/index.html',
        'comment': '', 
    }
    
    
    # Create list of variables
    variables = {}
    for vname in dim_database:
        data = getattr(m,vname)
        if 'time' in dim_database[vname]:
            variables[vname] = (dim_database[vname],data[np.newaxis,...])
        else:
            variables[vname] = (dim_database[vname],data)

    # Create list of coordinates
    coordinates = {}
    cname_to_1D = ['time','z','x','y','l','k']
    coords_1D = [(np.array([m.t])), (np.arange(1,m.nz+1)), (m.x[0,:]), (m.y[:,0]), (m.l[:,0]), (m.k[0,:])] 
    for cname in coord_database:
        if cname in cname_to_1D:
            index = cname_to_1D.index(cname)
            coordinates[cname] = coords_1D[index]
        else:
            try:
                data = getattr(m,cname)
                coordinates[cname] = (coord_database[cname],data )
            except:
                pass
        
    # Define dataset
    ds = xr.Dataset(variables,
                    coords=coordinates,
                    attrs=ds_attr_database)
    
    # Assign attributes to coordinates
    for caname in coord_attr_database:
        if caname in ds.coords:
            ds.coords[caname].attrs = coord_attr_database[caname]

    # Assign attributes to variables
    for vaname in var_attr_database:
        if vaname in ds.data_vars:
            ds.data_vars[vaname].attrs = var_attr_database[vaname]

            
    return ds