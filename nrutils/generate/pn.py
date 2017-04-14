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

# Calculate the Frequency domain amplitude of PN hlm Multipoles
def hf_amp_np( f,eta,X1z,X2z,mode ):
    '''
    Notes:
    * all inputs except mode must be float
    * mode must be iterable of length 2 containing spherical harmonic la and m
    '''
    # Import Useful things
    from numpy import sqrt,pi
    #
    x = (pi*f)**(2.0/3.0)   # This is sometimes called v

    # eq 6.4 of https://arxiv.org/pdf/1204.1043.pdf
    Atime = 8 * eta * x * sqrt(pi/5) * spa_intermediate_amplitude( x, eta,X1z,X2z,mode )

    #
    ans = abs( sqrt(pi) * Atime / sqrt( 1.5*sqrt(x)*XdotT4(x,eta,X1z,X2z) ) )
    
    #
    return ans

#
def spa_intermediate_amplitude( x,eta,X1,X2,mode ):

    #
    from numpy import sqrt,pi
    from nrutils.formula.HPN050317 import H

    #
    Delta   = -sqrt( 1-4*eta )  # NOTE the minus sign!
    Xa      = 0.5 * ( X1-X2 )
    Xs      = 0.5 * ( X1+X2 )

    #
    l,m = mode
    ans = H[l,m]( x,eta,Delta,Xa,Xs,X1,X2 )

    #
    return ans

#
def XdotT4(x,eta,X1,X2):
    #
    from nrutils.formula.HPN050317 import __XdotT4__
    from numpy import sqrt
    #
    # Asuming m1+m2=M=1 and m2>=m1
    M=1.0
    beta = sqrt(1.0-4.0*eta)
    m1=0.5*( M - beta * M)
    m2=0.5*( M + beta * M)
    #
    return __XdotT4__( x, m1, m2, eta, X1, X2 )
