import unittest
import numpy as np
import pyqg
import pytest

def test_derivatives():
    m = pyqg.QGModel()

    updown = lambda x: x if x < 32 else 64 - x
    linear_field = np.array([[updown(i) + updown(j) for i in range(64)] for j in range(64)]).astype(m.q.dtype)
    quadratic_field = linear_field ** 2

    m.q = np.array([linear_field, quadratic_field])

    fe = pyqg.FeatureExtractor(m)

    dq_dx = fe.extract_feature('ddx(q)')
    dq_dy = fe.extract_feature('ddy(q)')

    for i in range(64):
        # Derivatives of linear layer are all near-ish 1 (won't be exact due to FFT)
        assert dq_dx[0][i][1:31].max() * m.dx < 1.5
        assert dq_dx[0][i][1:31].min() * m.dx > 0.5
        assert dq_dy[0][:,i][33:63].max() * m.dx < -0.5
        assert dq_dy[0][:,i][33:63].min() * m.dx > -1.5

        # Derivatives of quadratic layer are all near-ish 2x or 2y
        assert (dq_dx[1]/m.q[0])[i][2:30].max() * m.dx < 2.5
        assert (dq_dx[1]/m.q[0])[i][2:30].min() * m.dx > 1.5
        assert (dq_dy[1]/m.q[0])[:,i][34:62].max() * m.dx < -1.5
        assert (dq_dy[1]/m.q[0])[:,i][34:62].min() * m.dx > -2.5


def test_model_and_model_dataset_equivalence():
    m = pyqg.QGModel()
    m._step_forward()

    fe1 = pyqg.FeatureExtractor(m)
    fe2 = pyqg.FeatureExtractor(m.to_dataset())

    np.testing.assert_allclose(fe1.extract_feature('laplacian(advected(q))'),
                               fe2.extract_feature('laplacian(advected(q))').data[0])
