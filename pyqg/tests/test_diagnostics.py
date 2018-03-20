from __future__ import print_function
from future.utils import iteritems
import unittest
import numpy as np
import pyqg
from pyqg import diagnostic_tools as diag

class DiagnosticsTester(unittest.TestCase):

    def test_describe_diagnostics(self):
        """ Test whether describe_diagnostics runs without error """

        m = pyqg.QGModel(1)
        m.describe_diagnostics()

if __name__ == "__main__":
    unittest.main()
