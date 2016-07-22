import numpy as np
from . import _solver_nfw_pywrap

class Solver(object):
    """Docstring."""

    def __init__(self, r0, beta, ucounts, bcounts, r, w, 
                        cpars=np.zeros(4,dtype='f8'), rsig=0.0):

        # ensure all correct length, etc.

        # we'll do that here in Python, and save the c code.
        # Though of course that makes the c code more fragile,
        # but it should always be accessed through here.
        
        self.r0 = float(r0)
        self.beta = float(beta)

        self.ucounts = ucounts.astype('f8')
        self.bcounts = bcounts.astype('f8')
        self.r = r.astype('f8')
        self.w = w.astype('f8')
        self.cpars = cpars.astype('f8')
        self.rsig = float(rsig)

        ngal = self.ucounts.size
        if (ngal != self.bcounts.size):
            raise ValueError("ucounts and bcounts must be same length")
        if (ngal != self.r.size):
            raise ValueError("ucounts and r must be the same length")
        if (ngal != self.w.size):
            raise ValueError("ucounts and w must be the same length")

        self._solver = _solver_nfw_pywrap.Solver(self.r0,
                                                 self.beta,
                                                 self.ucounts,
                                                 self.bcounts,
                                                 self.r,
                                                 self.w,
                                                 self.cpars,
                                                 self.rsig)

    def solve_nfw(self):
        return self._solver.solf_nfw()

        

                                                 
