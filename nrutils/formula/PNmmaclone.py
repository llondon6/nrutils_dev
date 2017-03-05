

#
def FAmpPN( f,eta,X1,X2,mode ):
    '''
    Notes:
    * all inputs except mode must be float
    * mode must be iterable of length 2 containing spherical harmonic la and m
    '''
    # Import Useful things
    from numpy import sqrt,pi
    #
    x = (pi*f)**(2.0/3.0)
    #
    Atime = 8 * eta * x * sqrt(pi/5) * PNAmplitude( x, eta,X1,X2,mode )
    #
    ans = abs( sqrt(pi) * Atime / sqrt( 1.5*sqrt(x)*XdotT4(x,eta,X1,X2) ) )
    #
    return ans

#
def PNAmplitude( x,eta,X1,X2,mode ):

    #
    from HPN050317 import H

    #
    Delta   = -sqrt( 1-4*eta )  # NOTE the minus sign!
    Xa      = 0.5 * ( X1-X2 )
    Xs      = 0.5 * ( X1+X2 )
    x       = (pi*f) ** (2/3)   # This is sometimes called v

    #
    l,m = mode
    ams = H[l,m]( x,eta,Delta,Xa,Xs,X1,X2 )

    #
    return None

#
def XdotT4(x,eta,X1,X2):
    #
    from HPN050317 import __XdotT4__
    from numpy import sqrt
    #
    # Asuming m1+m2=M=1 and m2>=m1
    M=1.0
    beta = sqrt(1.0-4.0*eta)
    m1=0.5*( M - beta * M)
    m2=0.5*( M + beta * M)
    #
    return __XdotT4__( x, m1, m2, eta, X1, X2 )
