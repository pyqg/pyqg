class PYQGError(Exception):
    """Base class for all PYQG exceptions"""
    pass

class DiagnosticNotFilledError(PYQGError):
    """Raise this when trying access a diagnostic that has not been filled"""
    def __init__(self, dname):
        self.dname = dname
        super().__init__(self.dname)
    
        pass