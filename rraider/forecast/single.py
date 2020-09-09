import pandas as pd
import numpy as np



class AssetModel:
	'''Base class from which all price models derive

	This class is used to standardize asset price assumptions,
	with methods to forecast prior assumptions into the future.

	In this way, after some features of a set of assets is estimated,
	models can create projected price distributions from monte carlo simulations.

	This might be useful in multi-asset baskets, where correlation between assets
	can make-or-break a set of trades.

	'''

	def simulate(self, horizon = 1, n:int = None, path:bool=False):
		'''Simulates future prices at some point in the future.

		Arguments:
			horizon:
				periods in the future to estimate
			n:
				number of trials
			path:
				returns tuple (x,y) where x is the time index and y has shape (nSims, pathLength)

		Returns:
			tuple x,y if path is true, otherwise np array of shape (nSims,)

		Example:
			.. code-block:: python

			   m = thisClass(S0 = 100, mu=0, sigma=0.2, N=1000)
			   x, y = m.distribution(horizon=3)

			   # Plot the data
			   plt.plot(x, y[0,:])
			   plt.show()
		'''
		return np.zeros(n,horizon)


	def distribution(self, horizon = 1):
		'''Finds the estimated price distribution at some point in the future.

		Arguments:
			horizon:
				the number of periods in the future to estimate

		Returns:
			x,y, where x are endpoints of histogram bins, and y is probability of lying between bins

		Example:

			.. code-block:: python

			   m = thisClass(S0 = 100, mu=0, sigma=0.2, N=1000)
			   x, dist = m.distribution(horizon=3)

			   # This plots the histogram correctly
			   plt.bar((x[1:]+x[:-1])/2.0,dist, width=x[1] - x[0])
			   plt.show()

		'''
		n = 1000
		sim = self.simulate( horizon=horizon, n=n, path=False)
		y, x = np.histogram(sim)
		y = y * 1.0 / n
		return x,y






class BrownianModel (AssetModel):
	def __init__ ( self, S0, mu, sigma, N ):
		"""Simulates brownian asset motion.
		
		Arguments:
			S0: Position at t=0
			mu: werner process drift, per T=1
			sigma: werner process stdev, per T=1
			N: simulation Resolution per unit of T
		"""
		self._N = N
		self._dt = self._N**-1
		self._S0 = S0
		self._mu = mu
		self._sigma = sigma


	def simulate(self, horizon:int = 1, n:int = 100, path:bool=False):
		T = float(horizon)
		pl = int(np.ceil(T//self._dt))
		S = (n,pl)

		deltas = (self._mu / self._N) + (self._dt**0.5) * self._sigma * np.random.normal (size=S)
		
		noise = np.concatenate(
			(
				np.zeros( (n,) ).reshape((-1,1)),
				deltas.cumsum(axis=1)
			),
			axis=1
		)
		y = self._S0 + noise
		if path:
			x = np.linspace(0,T,pl+1)
			return x, y
		else:
			return y[:,-1]
	
	def distribution(self, *args, **kwargs ):
		super().distribution(*args, **kwargs)
	


class LogBrownianModel (BrownianModel):
	def __init(self, *args, **kwargs):
		"""Simulates brownian asset motion.
		
		Arguments:
			S0: Position at t=0
			mu: drift parameter, per T=1
			sigma: underlying werner process volatility, per T=1
			N: simulation Resolution per unit of T
		"""
		super().__init__(*args, **kwargs)

		

	def simulate(self, horizon:int = 1, n:int = 100, path:bool=False):

		T = float(horizon)
		pl = int(np.ceil(T//self._dt))
		S = (n,pl)

		e = np.exp(
			(self._mu - 0.5 * self._sigma ** 2) * self._dt +
			(self._dt ** 0.5) * self._sigma * np.random.normal(size=S)
		)
		
		noise = np.concatenate(
			(
				np.ones( (n,) ).reshape((-1,1)),
				e.cumprod(axis=1),
			),
			axis=1
		)
		y = self._S0 * noise
    
		if path:
			x = np.linspace(0,T,pl+1)
			return x, y
		else:
			return y[:,-1]



	def distribution(self, *args, **kwargs ):
		super().distribution(*args, **kwargs)






class GarchModel (AssetModel):
    def __init__(self, returns, **kwargs ):
        '''Creates GARCH volatility model.
        
        See the documentation for `ARCH package`_ for full arguments
        
        Arguments:
            returns:
                numpy array of prices
        
        .. _`ARCH package`: https://arch.readthedocs.io/en/latest/api.html
        '''
        from arch import arch_model
        
        self.px = returns.iloc[-1]
        returns = returns.pct_change().dropna()
        
        self._model = arch_model(
            # Scale up the returns, so GARCH model can converge more easily
            returns*100,
            vol = 'Garch',
            rescale = False,
            **kwargs,
        )
        self._res = self._model.fit(disp='off')
        
        
    @property
    def summary(self):
        return self._res.summary()
    
    
    def simulate(self, horizon=1, n=1000, path=False):
        '''Simulates the asset distribution (as multiple of last price) at time ``horizon`` in the future.
        
        Arguments:
            horizon:
                Number of time periods to look forward
        
        Returns:
            numpy array, of shape (m,) 
        '''
        sims = self._res.forecast(
            horizon = horizon,
            method = 'simulation',
            simulations = n,
        ).simulations.values[-1,:,:] / 100.0
        results = np.cumprod(1+sims, axis=1) * self.px
        if path:
            x = np.arange(0,horizon)
            return x, results
        else:
            return results[:,-1].flatten()
