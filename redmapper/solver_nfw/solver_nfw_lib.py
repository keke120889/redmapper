"""Class to implement the zero-finding algorithm with nfw weights.
"""

from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
from . import _solver_nfw_pywrap

class Solver(object):
    """
    Class for the NFW Solver object.  This uses a zero-finding algorithm
    to solve for lambda, r_lambda, and membership probabilities, including
    softening parameters in radius and luminosity.  It must implement
    the nfw filter because the radius and hence radius weighting depends
    on richness and membership probabilities.
    """

    def __init__(self, r0, beta, ucounts, bcounts, r, w,
                        cpars=np.zeros(4,dtype='f8'), rsig=0.0):
        """
        Instantiate a Solver object.

        Parameters
        ----------
        r0: `float`
           Normalization of the radius-richness relation (Mpc)
        beta: `float`
           Power-law slope of the radius-richness relation.
        ucounts: `np.array`
           Float array of u(x) for the cluster neighbors.
        bcounts: `np.array`
           Float array of b(x) for the cluster neighbors.
        r: `np.array`
           Float array of radii for the cluster neighbors (Mpc).
        w: `np.array`
           Float array of theta_i * p_free weights
        cpars: `np.array`, optional
           4 element array of masking correction factor polynomial
           parameters.  Default is 0s (no mask correction).
        rsig: `float`, optional
           Radial softening parameter for theta_r(r).  Default is 0.0
           (no softening).
        """

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
        """
        Solve for the radius/richness/pmem using the nfw weights.

        Returns
        -------
        lambda: `np.array`
           Float array (1 element) with richness lambda
        p: `np.array`
           Float array with raw membership probabilities (no theta_i, theta_r,
           pfree) for neighbors.
        wt: `np.array`
           Float array with total membership probabilities
           (p * theta_r * theta_i * pfree) for neighbors.
        r_lambda: `np.array`
           Float array (1 element) with r_lambda radius
        theta_r: `np.array`
           Float array with theta_r(r) for neighbors.
        """
        return self._solver.solve_nfw()

