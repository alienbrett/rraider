from .single import AssetModel
from ..utils import *
import numpy as np
import pandas as pd
from .. import bs

import scipy.interpolate
import scipy.integrate
from scipy.ndimage import gaussian_filter1d











class ImpliedOptionModel (AssetModel):

	dayCount = 252

	def __init__(self, strikes, impVol, bid, ask, callput, time, irate, spot, volSmoother=None ):
		'''Initializes the implied option model.

		Arguments:
			strikes:
				numpy array or pandas series of strike prices
			impVol:
				numpy array or pandas series of implied option volatilities
			bid:
				numpy array or pandas series of bid
			ask:
				numpy array or pandas series of ask
			callput:
				'call' or 'put'
			time:
				numpy array or pandas series of time to expiration, a float
			irate:
				risk free interest rate
			spot:
				spot px
		'''

		self.df = pd.DataFrame(
			{
				'strike': strikes,
				'imp_vol': impVol,
				'price': (bid + ask) / 2.0,
				'bid': bid,
				'ask': ask,
				't': time,
			}
		)
		self.r = irate
		self.px = np.asarray(spot)
		self.callput = callput

		self.strikes = sorted(list(set(self.df['strike'].values)))
		self.times = sorted(list(set(self.df['t'].values)))

		if volSmoother is not None:
			self.setVolSmoother(volSmoother)
		else:
			self.setVolSmoother(
			   lambda vol: scipy.ndimage.gaussian_filter1d ( vol, sigma=4.0 )
			)


	def setVolSmoother(self, f):
		'''
		'''
		self._vf = f



	def _timePoint(self, t):
		return self.df.loc[ self.df['t'] == t ].sort_values('strike')
	


	def simulate(self, horizon = 1, n:int = None):
		x, cumProb = self._cdf ( horizon )

		percentiles = np.random.uniform(size=n)
		# print(percentiles)
		f = scipy.interpolate.interp1d(
			cumProb,
			x
		)
		strikes = f(percentiles)
		return strikes



	def _cdf(self, horizon, dividend_yield=0):


		dx = self._timePoint(horizon)

		x, prob = self.hypotheticalDist(
			horizon,
			self._vf ( dx['imp_vol'] ),
		)

		cumProb = scipy.integrate.cumtrapz (
			prob, x, initial=0
		)

		return x, cumProb


		


	def hypotheticalDist (self, horizon, vols, dividend_yield=0):
		'''Finds the implied distribution at a given time

		Arguments:
			horizon:
				time to expiration, used to grab relevent option records
			vols:
				hypothetical volumes used for pricing
			dividend_yield:
				hypothetical anualized yield

		Returns:

		'''
		import scipy.integrate

		dx = self._timePoint(horizon)

		# Calculate the implied 
		yhat, _ = bs.BlackScholes(
			spot_price = self.px,
			strike_price = dx['strike'],
			time_to_maturity = dx['t'],
			interest_rate = self.r,
			sigma = vols,
			dividend_yield = dividend_yield,
		).european_option_price()

		# Name change
		x = dx['strike']

		# Get the second derivatives
		x2, dy2 = derivApprox(
			x.values.flatten(),
			yhat.flatten(),
			n=2
		)

		def filterCondition( f, x, y ):
			# Preserve only positive values
			z = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1).T
			z = np.extract(
				np.tile(
					f(z[1,:]),
					reps=(2,1)
				),
				z
			).reshape(2,-1)
			return z[0,:], z[1,:]

		x2, dy2 = filterCondition(
			lambda y: y >= 0,
			x2, dy2
		)
		# print(x2.shape)
		# print(dy2.shape)

		total = scipy.integrate.trapz( dy2, x=x2 )
		# print(total)
		dy2 = dy2 / total

		return (x2, dy2)


	
