import pandas as pd
import numpy as np



class AssetModel:
	'''Base class from which all price models derive
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
		'''
		return np.zeros(n,horizon)


	def distribution(self, horizon = 1):
		'''Finds the estimated price distribution at some point in the future.

		Arguments:
			horizon:
				the number of periods in the future to estimate

		Returns:
			x,y, where x are endpoints of histogram bins, and y is probability of lying between bins
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
