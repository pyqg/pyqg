import numpy as np
from abc import ABC, abstractmethod

class Parameterization(ABC):
    """A generic class representing a subgrid parameterization. Inherit from
    this class, :math:`UVParameterization`, or :math:`QParameterization` to
    define a new parameterization."""

    @abstractmethod
    def __call__(self, m):
        r"""Call the parameterization given a pyqg.Model. Override this
        function in the subclass when defining a new parameterization. 

        Parameters
        ----------
        m : Model
            The model for which we are evaluating the parameterization.

        Returns
        -------
        forcing : real array or tuple
            The forcing associated with the model. If the model has been
            initialized with this parameterization as its
            :code:`q_parameterization`, this should be an array of shape
            :code:`(nz, ny, nx)`. For :code:`uv_parameterization`, this should
            be a tuple of two such arrays or a single array of shape :code:`(2,
            nz, ny, nx)`.
        """
        pass

    @property
    @abstractmethod
    def parameterization_type(self):
        """Whether the parameterization applies to velocity (in which case this
        property should return :code:`"uv_parameterization"`) or potential
        vorticity (in which case this property should return
        :code:`"q_parameterization"`). If you inherit from
        :code:`UVParameterization` or :code:`QParameterization`, this will be
        defined automatically.

        Returns
        -------
        parameterization_type : string
            Either :code:`"uv_parameterization"` or
            :code:`"q_parameterization"`, depending on how the output should be
            interpreted.
        """
        pass

    def __add__(self, other):
        """Add two parameterizations (returning a new object).

        Parameters
        ----------
        other : Parameterization
            The parameterization to add to this one.

        Returns
        -------
        sum : Parameterization
            The sum of the two parameterizations.
        """
        return CompositeParameterization(self, other)

    def __mul__(self, constant):
        """Multiply a parameterization by a constant (returning a new object).

        Parameters
        ----------
        constant : number
            Multiplicative factor for scaling the parameterization.

        Returns
        -------
        product : Parameterization
            The parameterization times the constant.
        """
        return WeightedParameterization(self, constant)

    __rmul__ = __mul__


class CompositeParameterization(Parameterization):
    """A sum of multiple parameterizations. Used in
    Parameterization#__add__."""

    def __init__(self, *params):
        assert len(set(p.parameterization_type for p in params)) == 1, \
            "all parameterizations must target the same variable (uv or q)"
        self.params = params

    @property
    def parameterization_type(self):
        return self.params[0].parameterization_type

    def __call__(self, m):
        return np.sum([f(m) for f in self.params], axis=0)

    def __repr__(self):
        return f"CompositeParameterization{self.params}"

class WeightedParameterization(Parameterization):
    """A weighted parameterization. Used in Parameterization#__mul__."""

    def __init__(self, param, weight):
        self.param = param
        self.weight = weight

    @property
    def parameterization_type(self):
        return self.param.parameterization_type

    def __call__(self, m):
        return np.array(self.param(m)) * self.weight

    def __repr__(self):
        return f"{self.weight} * {self.param}"

class UVParameterization(Parameterization):
    """A generic class representing a subgrid parameterization in terms of
    velocity. Inherit from this to define a new velocity parameterization."""

    parameterization_type = 'uv_parameterization'

class QParameterization(Parameterization):
    """A generic class representing a subgrid parameterization in terms of
    potential vorticity. Inherit from this to define a new potential vorticity
    parameterization."""

    parameterization_type = 'q_parameterization'

class Smagorinsky(UVParameterization):
    r"""Velocity parameterization from `Smagorinsky 1963`_.

    This parameterization assumes that due to subgrid stress, there is an
    effective eddy viscosity

    .. math:: \nu = (C_S \Delta)^2 \sqrt{2(S_{x,x}^2 + S_{y,y}^2 + 2S_{x,y}^2)}

    which leads to updated velocity tendencies :math:`\Pi_{i}, i \in \{1,2\}`
    corresponding to :math:`x` and :math:`y` respectively (equation is the same
    in each layer):

    .. math:: \Pi_{i} = 2 \partial_i(\nu S_{i,i}) + \partial_{2-i}(\nu S_{i,2-i})

    where :math:`C_S` is a tunable Smagorinsky constant, :math:`\Delta` is the
    grid spacing, and

    .. math:: S_{i,j} = \frac{1}{2}(\partial_i \mathbf{u}_j
                                  + \partial_j \mathbf{u}_i)

    .. _Smagorinsky 1963: https://doi.org/10.1175/1520-0493(1963)091%3C0099:GCEWTP%3E2.3.CO;2
    """

    def __init__(self, constant=0.1):
        r"""
        Parameters
        ----------
        constant : number
            Smagorinsky constant :math:`C_S`. Defaults to 0.1.
        """

        self.constant = constant

    def __call__(self, m, just_viscosity=False):
        r"""
        Parameters
        ----------
        m : Model
            The model for which we are evaluating the parameterization.
        just_viscosity : bool
            Whether to just return the eddy viscosity (e.g. for use in a
            different parameterization which assumes a Smagorinsky dissipation
            model). Defaults to false.
        """
        uh = m.fft(m.u)
        vh = m.fft(m.v)
        Sxx = m.ifft(uh * m.ik)
        Syy = m.ifft(vh * m.il)
        Sxy = 0.5 * m.ifft(uh * m.il + vh * m.ik)
        nu = (self.constant * m.dx)**2 * np.sqrt(2 * (Sxx**2 + Syy**2 + 2 * Sxy**2))
        if just_viscosity:
            return nu
        nu_Sxxh = m.fft(nu * Sxx)
        nu_Sxyh = m.fft(nu * Sxy)
        nu_Syyh = m.fft(nu * Syy)
        du = 2 * (m.ifft(nu_Sxxh * m.ik) + m.ifft(nu_Sxyh * m.il))
        dv = 2 * (m.ifft(nu_Sxyh * m.ik) + m.ifft(nu_Syyh * m.il))
        return du, dv

    def __repr__(self):
        return f"Smagorinsky(Cs={self.constant})"

class BackscatterBiharmonic(QParameterization):
    r"""PV parameterization based on `Jansen and Held 2014`_ and
    `Jansen et al.  2015`_ (adapted by Pavel Perezhogin). Assumes that a
    configurable fraction of Smagorinsky dissipation is scattered back to
    larger scales in an energetically consistent way.

    .. _Jansen and Held 2014: https://doi.org/10.1016/j.ocemod.2014.06.002
    .. _Jansen et al. 2015: https://doi.org/10.1016/j.ocemod.2015.05.007
    """

    def __init__(self, smag_constant=0.08, back_constant=0.99, eps=1e-32):
        r"""
        Parameters
        ----------
        smag_constant : number
            Smagorinsky constant :math:`C_S` for the dissipative model.
            Defaults to 0.08.
        back_constant : number
            Backscatter constant :math:`C_B` describing the fraction of
            Smagorinsky-dissipated energy which should be scattered back to
            larger scales. Defaults to 0.99. Normally should be less than 1,
            but larger values may still be stable, e.g. due to additional
            dissipation in the model from numerical filtering.
        eps : number
            Small constant to add to the denominator of the backscatter formula
            to prevent division by zero errors. Defaults to 1e-32.
        """

        self.smagorinsky = Smagorinsky(smag_constant)
        self.back_constant = back_constant
        self.eps = eps

    def __call__(self, m):
        lap = m.ik**2 + m.il**2
        psi = m.ifft(m.ph)
        lap_lap_psi = m.ifft(lap**2 * m.ph)
        dissipation = -m.ifft(lap * m.fft(lap_lap_psi * m.dx**2 * self.smagorinsky(m,
            just_viscosity=True)))
        backscatter = -self.back_constant * lap_lap_psi * (
            (np.sum(m.Hi * np.mean(psi * dissipation, axis=(-1,-2)))) /
            (np.sum(m.Hi * np.mean(psi * lap_lap_psi, axis=(-1,-2))) + self.eps)) 
        dq = dissipation + backscatter
        return dq

    def __repr__(self):
        return f"BackscatterBiharmonic(Cs={self.smagorinsky.constant}, "\
                                     f"Cb={self.back_constant})"

class ZannaBolton2020(UVParameterization):
    r"""Velocity parameterization derived from equation discovery by `Zanna and
    Bolton 2020`_ (Eq. 6).

    .. _Zanna and Bolton 2020: https://doi.org/10.1029/2020GL088376
    """

    def __init__(self, constant=-46761284):
        r"""
        Parameters
        ----------
        constant : number
            Scaling constant :math:`\kappa_{BC}`. Units: meters :sup:`-2`.
            Defaults to :math:`\approx -4.68 \times 10^7`, a value obtained by
            empirically minimizing squared error with respect to the subgrid
            forcing that results from applying the filtering method of `Guan et
            al. 2022`_ to a
            two-layer QGModel with default parameters.

            .. _Guan et al. 2022: https://doi.org/10.1016/j.jcp.2022.111090
        """

        self.constant = constant

    def __call__(self, m):
        # Compute ZB2020 basis functions
        uh = m.fft(m.u)
        vh = m.fft(m.v)
        vx = m.ifft(vh * m.ik)
        vy = m.ifft(vh * m.il)
        ux = m.ifft(uh * m.ik)
        uy = m.ifft(uh * m.il)
        rel_vort = vx - uy
        shearing = vx + uy
        stretching = ux - vy
        # Combine them in real space and take their FFT
        rv_stretch = m.fft(rel_vort * stretching)
        rv_shear = m.fft(rel_vort * shearing)
        sum_sqs = m.fft(rel_vort**2 + shearing**2 + stretching**2) / 2.0
        # Take spectral-space derivatives and multiply by the scaling factor
        kappa = self.constant
        du = kappa * m.ifft(m.ik*(sum_sqs - rv_shear) + m.il*rv_stretch)
        dv = kappa * m.ifft(m.il*(sum_sqs + rv_shear) + m.ik*rv_stretch)
        return du, dv

    def __repr__(self):
        return f"ZannaBolton2020(κ={self.constant:.2e})"

def Gaussian_Filter(w, m, FGR):
    w_fft = m.fft(w)
    Delta = m.dx * FGR
    k2 = m.k**2 + m.l**2
    G = np.exp(-k2*Delta**2/24)
    return m.ifft(G * w_fft)

def deconvolve(w, m, FGR, order):
    wd = w
    r = w
    for j in range(order):
        r = r - Gaussian_Filter(r, m, FGR)
        wd = wd + r
    return wd

def SFS(q, u, v, m, FGR):
    qf = Gaussian_Filter(q, m, FGR)
    uf = Gaussian_Filter(u, m, FGR)
    vf = Gaussian_Filter(v, m, FGR)
    quf = Gaussian_Filter(q*u, m, FGR)
    qvf = Gaussian_Filter(q*v, m, FGR)
    SFSu = qf*uf - quf
    SFSv = qf*vf - qvf
    return SFSu, SFSv 

class ADM(QParameterization):
    def __init__(self, FGR=2, order=5, skip=1, MSE=0.0):
        self.FGR = FGR
        self.order = order
        self.skip = skip
        self.MSE = MSE
    def __call__(self, m):
        if m.tc % self.skip == 0:
            qd = deconvolve(m.q, m, self.FGR, self.order)
            ud = deconvolve(m.u, m, self.FGR, self.order)
            vd = deconvolve(m.v, m, self.FGR, self.order)
            
            SFSu, SFSv = SFS(qd, ud, vd, m, self.FGR)
            
            ik = m.k * 1j
            il = m.l * 1j
            
            real = lambda q: q if q.shape == m.q.shape else m.ifft(q)
            spec = lambda q: q if q.shape != m.q.shape else m.fft(q)
            ddx = lambda q: real(ik * spec(q))
            ddy = lambda q: real(il * spec(q))

            dq = ddx(SFSu) + ddy(SFSv)

            var_z = np.var(dq, axis=(1,2))[:,np.newaxis,np.newaxis]
            noise = np.sqrt(self.MSE * var_z) * np.random.randn(*dq.shape)
            dq = dq + noise
            
            self.dq = dq
        else:
            try:
                dq = self.dq
            except:
                dq = m.q * 0
        
        return dq

    def __repr__(self):
        return f'ADM, FGR={self.FGR}, skip={self.skip}'

class Reynolds_stress(QParameterization):
    def __init__(self, FGR=2, Csim=12):
        self.FGR = FGR
        self.Csim = Csim 
    
    def __call__(self, m):
        qr = m.q - Gaussian_Filter(m.q, m, self.FGR)
        ur = m.u - Gaussian_Filter(m.u, m, self.FGR)
        vr = m.v - Gaussian_Filter(m.v, m, self.FGR)
        
        SFSu, SFSv = SFS(qr, ur, vr, m, self.FGR)
        
        ik = m.k * 1j
        il = m.l * 1j
        
        real = lambda q: q if q.shape == m.q.shape else m.ifft(q)
        spec = lambda q: q if q.shape != m.q.shape else m.fft(q)
        ddx = lambda q: real(ik * spec(q))
        ddy = lambda q: real(il * spec(q))
        
        return self.Csim*(ddx(SFSu) + ddy(SFSv))

    def __repr__(self):
        return f'Reynolds, FGR={self.FGR}, Csim={self.Csim}'