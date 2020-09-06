import ally
import numpy as np
import pandas as pd
from . import simulate, estimate, utils
from datetime import datetime, timedelta

from joblib import Memory

cachedir = '/tmp/joblib'
memory = Memory(cachedir, verbose=False)
a = ally.Ally()



@memory.cache
def getSymbol (sym, today, yesterday):
	return a.timesales (
		symbols = sym,
		startdate = yesterday,
		enddate = today,
		interval = '1min'
	)



class BracketPosition:
	def __init__ ( self, sym, direction, T=1,
				  timespan='day', volEWM = True, N=2*1000, M=5*1000, mu=0.01/252, r=0.01,
				  lookback = 2, miDay=390 ):
		"""Computes bracket parameters to meet certain risk criteria.
		
		Arguments:
			sym:
				0
		"""
		
		self.d = direction
		
		# For day orders, risk-free return on daily basis is mean of log-brownian
		mu = (1+r)**(1/252) - 1
		
		# Lookback for timesales
		today	 = datetime.strftime(datetime.now(),'%Y-%m-%d')
		yesterday = datetime.strftime(datetime.now() - timedelta(days=2),'%Y-%m-%d')
		df		= getSymbol(sym, today, yesterday)

		# Classic vanilla volatility estimate
		self.vol = utils.volAdjust(
			estimate.volVanilla( df['last'], ewm=volEWM ),
			nPeriods = miDay # minutes in trading day
		)
		print('daily volatility estimate: {:.1f}%'.format(100*self.vol))
		
		quote = a.quote(sym, fields=['bid','ask'])
		# The entry price of instrument
		self.price = (quote['ask'] if self.d == 'bull' else quote['bid']).values[0]
		self.spread = (quote['ask'] - quote['bid']).values
		
		# Now simulate values
		self.sim = simulate.geomBrownianSim(
			S0 = self.price,
			mu = mu,
			sigma = self.vol,
			T = T,
			N = N,
			M = M
		)
		
		# Save our estimators
		self.lowEst, self.highEst = simulate.fullExecEstimator(self.sim['y'], self.spread)
		
		
		
	def lossEst ( self, p, price=None ):
		'''Profit target corresponding to probability, in $'s offset from the current price'''
		if price is None:
			price = self.price
			
		return ( self.lowEst(p) - price ) * (1.0 if self.d == 'bull' else -1.0)
	
	
	
	def profEst ( self, p, price=None ):
		'''Profit target corresponding to probability, in $'s offset from the current price'''
		if price is None:
			price = self.price
			
		return ( self.highEst(p) - price ) * (1.0 if self.d == 'bull' else -1.0)
	
	
	
	def shares ( self, dollarLoss, pctLoss, price=None ):
		'''Position size corresponding to the max $ loss allowable, and the %loss (and optionally current price)'''
		return np.floor ( dollarLoss / (self.lossEst(pctLoss, price=price) * -1.0) )
	


	def summarize ( self, dollarLoss, pctLoss, pctGain, price=None ):
		return {
			'lossExit': self.lossEst(pctLoss),
			'profExit': self.profEst(pctGain),
			'shares': self.shares(dollarLoss, pctLoss),
		}
	
	
