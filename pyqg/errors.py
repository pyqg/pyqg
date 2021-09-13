class PYQGError(Exception):
    """Base class for all PYQG exceptions"""
    pass

class DiagnosticNotFilledError(PYQGError):
    """Raise this when trying access a diagnostic that has not been filled"""
    def __init__(self, dname):
        message = f"Diagnostic {dname} has not been filled. Please adjust tavestart and taveint."
        super().__init__(message)
    pass