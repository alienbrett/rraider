from .single import AssetModel
from ..utils import mass_broadcast

import numpy as np


class MultiBrownianModel (AssetModel):
	def __init__ ( self, S0, mu, cov, N ):
		"""Simulates brownian asset motion on a universe of correlated assets
		
		Arguments:
			S0:
				Vector of asset prices at t=0
			mu:
				vector of werner process drifts, per T=1
			cov:
				covariance matrix
			N:
				simulation Resolution per unit of T
		"""
		self._N = N
		self._dt = self._N**-1

		# Store covariance
		self._cov = cov

		# Convert all inputs to the same standard vector size and shape
		self._S0, self._mu, _ = mass_broadcast( S0, mu, self._cov[0] )

		self._l = self._S0.shape[0]


	def simulate(self, horizon:int = 1, n:int = 100, path:bool=False):
		'''Simulates future prices at some point in the future.

		Arguments:
			horizon:
				periods in the future to estimate
			n:
				number of trials
			path:
				returns tuple (x,y) where x is the time index and y has shape (nSims, pathLength, nAssets)

		Returns:
			tuple x,y if path is true, otherwise np array of shape (nSims,nAssets)
		'''
		T = float(horizon)
		pl = int(np.ceil(T//self._dt))
		S = (n,pl)

		deltas = (self._mu.reshape(-1) / self._N) + \
			(self._dt**0.5) * \
			np.random.multivariate_normal (
				self._mu.reshape(-1),
				self._cov,
				S
			)
		
		j = np.zeros( (n, 1, self._l) )
		noise = np.concatenate(
			(
				j,
				deltas.cumsum(axis=1)
			),
			axis=1
		)
		y = self._S0.reshape(1,1,-1) + noise
		
		if path:
			x = np.linspace(0,T,pl+1)
			return x, y
		else:
			return y[:,-1,:]




class MultiLogBrownianModel (MultiBrownianModel):
	def __init(self, *args, **kwargs):
		"""Simulates geometric brownian asset motion on a universe of correlated assets
		
		Arguments:
			S0:
				Vector of asset prices at t=0
			mu:
				vector of werner process drifts, per T=1
			cov:
				covariance matrix
			N:
				simulation Resolution per unit of T
		"""
		super().__init__(*args, **kwargs)
		


	def simulate(self, horizon:int = 1, n:int = 100, path:bool=False):

		T = float(horizon)
		pl = int(np.ceil(T//self._dt))
		S = (n,pl)

		e = np.exp(
			(self._dt ** 0.5) * \
			np.random.multivariate_normal(
				self._mu.reshape(-1),
				self._cov,
				S
			) + \
			self._dt * self._mu.reshape(-1)
		)
		
		noise = np.concatenate(
			(
				np.ones( (n, 1, self._l) ),
				e.cumprod(axis=1),
			),
			axis=1
		)
		y = self._S0.reshape(1,1,-1) * noise

		if path:
			x = np.linspace(0,T,pl+1)
			return x, y
		else:
			return y[:,-1,:]
