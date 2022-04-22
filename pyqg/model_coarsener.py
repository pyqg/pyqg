class ModelCoarsener:
    def __init__(self, model, lower_resolution):
        self.m1 = model
        self.m2 = model.reinitialized(nx=lower_resolution, ny=lower_resolution)
        self.m2.q = self.coarsen(self.m1.qh)
        self.m2._invert()
        self.m2._calc_derived_fields()

        assert self.coarsening_ratio > 1

    @abstractmethod
    def coarsen(self, field):
        pass

    def subgrid_forcing(self, quantity):
        x1 = getattr(self.m1, quantity)
        x2 = getattr(self.m2, quantity)
        return (self.coarsen(self.m1._advect(x1)) -
                self.m2.ifft(self.m2._advect(x2)))
    
    @property
    def coarsening_ratio(self):
        return self.m1.nx / self.m2.nx

    @property
    def coarsened_model(self):
        return self.m2

class SpectralModelCoarsener(ModelCoarsener):
    def __init__(self, model, lower_resolution, filtr=lambda m: m.filtr):
        self.filtr = filtr
        super().__init__(model, lower_resolution)

    def coarsen(self, field):
        # Convert to spectral (if needed)
        if field.shape != self.m1.qh.shape:
            assert field.shape == self.m1.q.shape
            field = self.m1.fft(field)
        # Truncate high-frequency indices & filter
        nk = self.m2.qh.shape[1]//2
        truncated = np.hstack((field[:,:nk,:nk+1], field[:,-nk:,:nk+1]))
        filtered = truncated * self.filtr(self.m2) * self.coarsening_ratio**2
        # Convert to real
        return m2.ifft(filtered)

class RealspaceModelCoarsener(ModelCoarsener):
    def __init__(self, model, lower_resolution, filtr=None):
        if filtr is None:
            filtr = lambda v: np.ones_like(v)
        self.filtr = filtr
        super().__init__(model, lower_resolution)

    def coarsen(self, field):
        # Convert to real (if needed)
        if field.shape != self.m1.q.shape:
            assert field.shape == self.m1.qh.shape
            field = self.m1.ifft(field)
        field = self.filtr(field)
        
        assert self.coarsening_ratio == int(self.coarsening_ratio)
        pass

