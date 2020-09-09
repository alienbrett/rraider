from .single import AssetModel
from ..utils import *
import numpy as np
import pandas as pd

import scipy.interpolate











class ImpliedOptionModel (AssetModel):

	dayCount = 252

	def __init__(self, strikes, impVol, bid, ask, callput, time, irate, spot ):
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





	def volF (self, s, t):
		'''Evaluate volatility for a specific day

		 use only that day to compute values
		'''
		dx = self.df.loc[ self.df['t'] == t ]
		if len(dx) > 0:

			
			# f = scipy.interpolate.UnivariateSpline(
			# 	dx['strike'],
			# 	dx['imp_vol'],
			# )
			f = np.poly1d(
				np.polyfit(
					dx['strike'],
					dx['imp_vol'],
					deg=25
				)
			)

			return f(s)


		else:
			return None

	
	def distribution (self, horizon, d=2):
		'''Finds the implied distribution at a given time

		'''
		pass



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
		from .. import bs

		dx = self.df.loc[ self.df['t'] == horizon ]

		# Calculate the implied 
		c, _ = bs.BlackScholes(
			spot_price = self.px,
			strike_price = dx['strike'],
			time_to_maturity = dx['t'],
			interest_rate = self.r,
			sigma = vols,
			dividend_yield = dividend_yield,
		).european_option_price()

		# Get the relevent info
		x, yhat = dx['strike'], c

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

		# This converts second derivative into
		# the PDF
		modifier = np.exp(self.r * horizon)

		return (x2, modifier * dy2)


	
