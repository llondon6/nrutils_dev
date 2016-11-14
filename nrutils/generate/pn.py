# Module to calculate PN strain waveforms. The output will be members of the gwylm class.

'''
References

[1] "Comparison of post-Newtonian templates ... "
    Buonanno, Iyer, Ochsner et al
    https://arxiv.org/pdf/0907.0700.pdf
    ** NONSPINNING

[2] https://arxiv.org/pdf/1502.01747v1.pdf
    ** GENERAL SPINS

PLAN:

    Here will will first implement a non-precessing Taylor T4 PN approximant [1], and test it.

    Then we will implement a precessing PN aproximant and test it by comparing to Gerosa's PRECESSION package.

'''


# Import the core
from nrutils.core import *

# A class for the waveform generation workflow
class pnns:

    # Default options
    default_options = {
                        'fref' : 0.001,
                        'dt'   : 1e-3
                      }

    #
    def __init__(this,                        # The current object
                 options = default_options,   # Optional inputs
                 verbose = False ):           # Let the people know
        return None

    # ode solver
    def solve(this):
        #
        from scipy.integrate import ode

        #
        f = []

        # The PN jacobain
        jac = []

#
def pnns_t4_dvbydt():

    #
    ans = None

    return ans

# Separate modules for RHS? Different PN orders?
