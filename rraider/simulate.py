import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate

def brownianSim ( S0, mu, sigma, T, N, M, **kwargs ):
    """Simulates brownian motion over some interval.
    
    Arguments:
        S0: Position at t=0
        mu: werner process drift, per T=1
        sigma: werner process stdev, per T=1
        T: units of time to simulate
        N: simulation Resolution per unit of T
        M: trials to run
    """
    N = float(N)
    T = float(T)
    dt = N**-1
    n = 1+int(T//dt)
    S = (M,n)
    
    deltas = (mu / N) + (dt**0.5) * sigma * np.random.normal (size=S)
    
    x = np.linspace(0,T,n+1)
    noise = np.concatenate(
        (
            np.zeros( (M,) ).reshape((-1,1)),
            deltas.cumsum(axis=1)
        ),
        axis=1
    )
    y = S0 + noise
    return {
        'x': x,
        'y': y
    }
    
    
def geomBrownianSim ( S0, mu, sigma, T, N, M, **kwargs):
    """Simulates log-normal brownian motion."""
    N = float(N)
    T = float(T)
    dt = N**-1
    n = 1+int(T//dt)
    S = (M,n)
    
    e = np.exp(
        (mu - 0.5 * sigma**2) * dt +
        (dt**0.5) * sigma * np.random.normal (size=S)
    )
    
    x = np.linspace(0,T,n+1)
    noise = np.concatenate(
        (
            np.ones( (M,) ).reshape((-1,1)),
            e.cumprod(axis=1)
        ),
        axis=1
    )
    y = S0 * noise
    return {
        'x': x,
        'y': y
    }
    

    

    
def minsAndMaxs ( ys ):
    """Finds the global mins and maxes for a time series of prices.
    Be sure that y's have shape (nSims, timeSteps)
    """
    return np.min ( ys, axis=1 ), np.max ( ys, axis=1 )



def countToCDF ( ys, resolution=100, forward=False ):
    """Turn set of counts into a PDF.
    Let ys have the shape (n,)
    forward argument to True specifies reverse-direction CDF
    """
    
    s = ys.shape[0]
    y = np.sort(ys)

    if forward:
        ys = np.flip(ys)
    
    raw = y
    
    integral = scipy.integrate.trapz (
        raw, 
        dx = 1.0/resolution
    )
    
    cumIntegral = np.concatenate (
        (
            scipy.integrate.cumtrapz (
                raw, 
                dx = 1.0/resolution
            ),
            []
        )
    )
    
    raw = raw[:-1]
    
    if forward:
        cumIntegral = np.flip( cumIntegral )

    return (cumIntegral / integral, raw)




def simpleExecEstimator ( simulations, **kwargs ):
    """Generate execution functions, assuming no bid/ask spread.
    
    Returns two functions, f(p) and g(p) that will return the price at which a
	stock has probability p of executing over timespan T.
    'f' describes price of buys (measuring minimums) and 'g' describes price of sells (measuring maximums).
    
    'Simulations' is a numpy matrix describing some set of simulations, with shape (n sims, path length).
    """
    kind = kwargs.get('kind', 'linear')
    mins, maxs = minsAndMaxs ( simulations )
    
    minCdf = countToCDF ( np.sort(mins), resolution = kwargs.get('resolution',10), forward=False )
    maxCdf = countToCDF ( np.sort(maxs), resolution = kwargs.get('resolution',10), forward=True )
    
    maxCdf = np.flip(maxCdf[0]), np.flip(maxCdf[1])
    
    f = scipy.interpolate.interp1d( minCdf[0], minCdf[1], kind=kind, fill_value='extrapolate', assume_sorted=True)
    g = scipy.interpolate.interp1d( maxCdf[0], maxCdf[1], kind=kind, fill_value='extrapolate', assume_sorted=True)
    
    return f,g



def fullExecEstimator ( simulations, spread, **kwargs ):
	"""Build estimator objects that predict the probablitity of order execution at given level,
	given bid/ask spread.
	"""
	halfSpread = abs(spread) / 2.0

	# The original f-prime, g-prime functions
	fp,gp = simpleExecEstimator ( simulations, **kwargs )
	
	# Add the spread
	f = lambda x: fp (x) + halfSpread
	g = lambda x: gp (x) - halfSpread

	return f,g

