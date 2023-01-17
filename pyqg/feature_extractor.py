import re
import operator
import numpy as np
import xarray as xr

class FeatureExtractor:
    """Helper class for taking spatial derivatives and translating string
    expressions into data. Works with either pyqg.Model or xarray.Dataset."""

    def __init__(self, model_or_dataset):
        self.m = model_or_dataset
        self.cache = {}

        if hasattr(self.m, '_ik'):
            self.ik, self.il = np.meshgrid(self.m._ik, self.m._il)
        elif hasattr(self.m, 'fft'):
            self.ik = 1j * self.m.k
            self.il = 1j * self.m.l
        else:
            k, l = np.meshgrid(self.m.k, self.m.l)
            self.ik = 1j * k
            self.il = 1j * l

        self.nx = self.ik.shape[0]
        self.wv2 = self.ik**2 + self.il**2

    def __call__(self, feature_or_features, flat=False):
        arr = lambda x: x.data if isinstance(x, xr.DataArray) else x
        if isinstance(feature_or_features, str):
            res = arr(self.extract_feature(feature_or_features))
            if flat: res = res.reshape(-1)

        else:
            res = np.array([arr(self.extract_feature(f)) for f in feature_or_features])
            if flat: res = res.reshape(len(feature_or_features), -1).T
        return res

    # Helpers for taking FFTs / deciding if we need to
    def fft(self, x):
        try:
            return self.m.fft(x)
        except:
            # Convert to data array
            dims = [dict(y='l',x='k').get(d,d) for d in self['q'].dims]
            coords = dict([(d, self[d]) for d in dims])
            return xr.DataArray(np.fft.rfftn(x, axes=(-2,-1)), dims=dims, coords=coords)

    def ifft(self, x):
        try:
            return self.m.ifft(x)
        except:
            return self['q']*0 + np.fft.irfftn(x, axes=(-2,-1))

    def is_real(self, arr):
        return len(set(arr.shape[-2:])) == 1

    def real(self, arr):
        arr = self[arr]
        if isinstance(arr, float): return arr
        if self.is_real(arr): return arr
        return self.ifft(arr)

    def compl(self, arr):
        arr = self[arr]
        if isinstance(arr, float): return arr
        if self.is_real(arr): return self.fft(arr)
        return arr

    # Spectral derivatrives
    def ddxh(self, f): return self.ik * self.compl(f)
    def ddyh(self, f): return self.il * self.compl(f)
    def divh(self, x, y): return self.ddxh(x) + self.ddyh(y)
    def curlh(self, x, y): return self.ddxh(y) - self.ddyh(x)
    def laplacianh(self, x): return self.wv2 * self.compl(x)
    def advectedh(self, x_):
        x = self.real(x_)
        return self.ddxh(x * self.m.ufull) + self.ddyh(x * self.m.vfull)

    # Real counterparts
    def ddx(self, f): return self.real(self.ddxh(f))
    def ddy(self, f): return self.real(self.ddyh(f))
    def laplacian(self, x): return self.real(self.laplacianh(x))
    def advected(self, x): return self.real(self.advectedh(x))
    def curl(self, x, y): return self.real(self.curlh(x,y))
    def div(self, x, y): return self.real(self.divh(x,y))

    # Main function: interpreting a string as a feature
    def extract_feature(self, feature):
        """Evaluate a string feature, e.g. laplacian(advected(curl(u,v)))."""

        # Helper to recurse on each side of an arity-2 expression
        def extract_pair(s):
            depth = 0
            for i, char in enumerate(s):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 0:
                    return self.extract_feature(s[:i].strip()), self.extract_feature(s[i+1:].strip())
            raise ValueError(f"string {s} is not a comma-separated pair")

        real_or_spectral = lambda arr: arr + [a+'h' for a in arr]

        if not self.extracted(feature):
            # Check if the feature looks like "function(expr1, expr2)"
            # (better would be to write a grammar + use a parser,
            # but this is a very simple DSL)
            match = re.search(f"^([a-z]+)\((.*)\)$", feature)
            if match:
                op, inner = match.group(1), match.group(2)
                if op in ['mul', 'add', 'sub', 'pow']:
                    self.cache[feature] = getattr(operator, op)(*extract_pair(inner))
                elif op in ['neg', 'abs']:
                    self.cache[feature] = getattr(operator, op)(self.extract_feature(inner))
                elif op in real_or_spectral(['div', 'curl']):
                    self.cache[feature] = getattr(self, op)(*extract_pair(inner))
                elif op in real_or_spectral(['ddx', 'ddy', 'advected', 'laplacian']):
                    self.cache[feature] = getattr(self, op)(self.extract_feature(inner))
                else:
                    raise ValueError(f"could not interpret {feature}")
            elif re.search(f"^[\-\d\.]+$", feature):
                # ensure numbers still work
                return float(feature)
            elif feature == 'streamfunction':
                # hack to make streamfunctions work in both datasets & pyqg.Models
                self.cache[feature] = self.ifft(self['ph'])
            else:
                raise ValueError(f"could not interpret {feature}")

        return self[feature]

    def extracted(self, key):
        return key in self.cache or hasattr(self.m, key)

    # A bit of additional hackery to allow for the reading of features or properties
    def __getitem__(self, q):
        if isinstance(q, str):
            if q in self.cache:
                return self.cache[q]
            elif re.search(f"^[\-\d\.]+$", q):
                return float(q)
            else:
                return getattr(self.m, q)
        elif any([isinstance(q, kls) for kls in [xr.DataArray, np.ndarray, int, float]]):
            return q
        else:
            raise KeyError(q)
