import numpy as np
import pandas as pd

__all__ = ['volParkinson', 'volVanilla']


    
    
def difs ( close ):
    return np.log(close).diff()



def volParkinson ( highs, lows ):
    """Estimates the historical volatility for series using the Parkinson method.

	Arguments:
		highs:
			numpy array or pandas series of daily high values
		lows:
			numpy array or pandas series of daily low values

	Returns:
		Parkinson volatility estimate
	
    """
    
    xt = np.log ( highs / lows )
    xt = xt**2 / (4 * np.log(2))
    
    res = np.mean(xt) ** 0.5
    return res


def volVanilla ( close, ewm:bool = False, halflife:float = 60 ):
    """Estimates the historical volatility for series using the standard 'close' method.

	Arguments:
		close:
			numpy array or pandas series of daily close values
		ewm:
			boolean, whether to use exponential-moving avererage, or to use flat variance
		halflife:
			float, rate parameter for ewm, if applicable

	Returns:
		Historical volatility estimate
	
    """
    x = pd.Series( difs(close), index=close.index)
    if ewm:
        x = x.ewm(halflife)
    j = x.std()
    return j.iloc[-1]
