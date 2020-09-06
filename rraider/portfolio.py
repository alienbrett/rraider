import yfinance as yf
import numpy as np
import pandas as pd
from . import utils


def ensureNormal ( df, **kwargs ):
	if kwargs.get('isLog',True):
		# print("transforming")
		dx = np.diff(np.log(df))
	else:
		# print('skipping')
		dx = df
	return dx


def basketVariance ( samples, weights, **kwargs ):
	"""Returns the variance a basket, weighted by 'weights'.
	
	'samples' should have shape (k,n) for k variables with n samples each.
	'weights' should be vector of shape (k,)
	"""
	# Covariance matrix
	c = np.cov ( ensureNormal(samples, **kwargs) )
	# Sum down to variance
	x = (weights.T @ c) @ weights.T
	return x
	
	
	
	
def minimumVariance ( samples, **kwargs ):
	"""Returns the weights that minimize the variance among all baskets.
	
	Let samples have shape (k,n) as usual. K variables, n samples.
	"""
	dx = ensureNormal(samples,**kwargs)
	cov = np.cov(dx)
	x = np.linalg.inv (
		cov
	) @ np.ones ( samples.shape[0] )
	return x / np.linalg.norm(x, ord=1)




def minVarHedgeRatio ( df1, df2, **kwargs ): 
	"""Computes a simple minimum-variance hedge ratio.
	
	Provide 2 pandas series, df1 & df2,
		and I will provide the ratio units allocated in  df2 to hedge each unit of df1
	"""
	df1 = ensureNormal(df1, **kwargs)
	df2 = ensureNormal(df2, **kwargs)
	
#	 print(df1.shape, df2.shape)
	
	y = np.concatenate( [df1.reshape(-1,1),df2.reshape(-1,1)], axis=1 )
#	 print(y.shape)
	x = minimumVariance ( y.T, isLog=False )
#	 print("Min var allocs:", x)
	r = x[1] / x[0]
#	 print("Hedge ratio found:",r)
	return r
	

	
def singleHedgeAnalyze ( df1, df2, **kwargs ):
	"""Given two pd.Series objects,
	compute the hedge ratio, and variance improvement information.
	Returns dictionary
	"""
	# Compute hedge ratio
	ratio = minVarHedgeRatio( df1, df2, **kwargs )
	leverage = 1 + ratio
	
	# Simulated portfolio weights
	wRaw = np.array([1.0, 0])
	wHdg = np.array([1.0, ratio])
	wHdgAdj = wHdg / np.abs(np.sum(wHdg))
	
	# Create portfolio universe, and test the variance
	dfx = np.concatenate([df1.reshape(-1,1),df2.reshape(-1,1)],axis=1).T
	
	# Variance of single-asset
	varRaw = basketVariance(dfx, wRaw, **kwargs)
	# Variance of portfolio with single asset,
	#  adding in the hedge asset
	varHdg = basketVariance(dfx, wHdg, **kwargs)
	# Variance of varHdg, but adjusted
	#  so position net size equals single-asset portfolio
	adjVarHdg = basketVariance(dfx, wHdgAdj, **kwargs)
	
	if kwargs.get('verbose',True):
		print(
			'{:.1f}% variance improvement'.format(
				100*(varRaw - varHdg) / varRaw
			)
		)
		print(
			'{:.1f}% variance improvement, for same-size portfolio'.format(
				100*(varRaw - adjVarHdg) / varRaw
			)
		)
	return {
		'hedge ratio': ratio,
		'leverage': leverage,
		'net': 'short' if leverage < 0 else 'long',
		'hedge': {
			'weights': wHdg,
			'variance': varHdg,
			'adjusted variance': adjVarHdg,
		},
		'unhedged': {
			'weights': wRaw,
			'variance': varRaw,
		},
	}




def basketHedgeAnalyze ( df, positions, **kwargs):
	"""Construct some additional set of positions to hedge a given set of positions.
	df -> pd.DataFrame, price timeseries
	positions-> pd.Series, index is position, value is size in $$$
	"""
	hdgSyms = [ x for x in df.columns if x not in positions.index ]
	
	# Stopping conditions
	if len(hdgSyms) == 0 or len(positions) == 0:
		return positions
	
	data = ensureNormal(df.values.T, **kwargs).T
	# index = df.index[:-1] if data.shape[1] != df.values.shape[1] else df.index
	df = pd.DataFrame(
		data = data,
		columns = df.columns,
		index = df.index[:data.shape[0]]
	)
	k = {
		'isLog': False,
		'verbose': kwargs.get('verbose'),
	}

	oldVar = basketVariance (
		df[positions.index.tolist()].values.T,
		positions.values,
		**k
	)
	print( 'old variance:', oldVar )
	
	netSize = np.abs(positions.sum())

	# Get the % allocation in each symbol
	normalWeights = positions.values.reshape(-1,1) / netSize
	
	portfolio =  (df[positions.index.tolist()].values @ normalWeights).flatten()
	
	# Portfolio net pct-change
	portfolio = pd.Series(
		portfolio,
		index = df.index
	)
	
	res = []
	for x in hdgSyms:
		j = singleHedgeAnalyze(
			portfolio.values,
			df[x].values,
			isLog=False,
			verbose = kwargs.get('verbose',False)
		)
		j['symbol'] = x
		res.append(j)
	
	# Get the top value on top
	res.sort(key=lambda x: x['hedge']['adjusted variance'])
	# print(res[:2])

	p = positions.append(
		pd.Series(
			[res[0]['hedge ratio'] * netSize],
			index=[res[0]['symbol']]
		)
	)
	print(p)
	print('net long: ${:.2f}'.format( p.sum() ))
	newVar = basketVariance (
		df[p.index.tolist()].values.T,
		p.values,
		**k
	)
	print( 'new variance:', newVar )
	
	
	if oldVar > newVar:
		print("We see improvement. Adding...")
		return basketHedgeAnalyze (df, p, **k)
	else:
		print("No improvement. Terminating.")
		return positions




def allocSimulator ( sym, N=1000, period='1y', interval='1d', riskFree=0.01 ):

	df = yf.download(
		tickers=sym,
		period=period,
		interval=interval,
		auto_adjust=True
	)['Close'][sym]

	dx = ensureNormal ( df.values.T, isLog = True ).T
	df = pd.DataFrame(
		data = dx,
		columns = df.columns,
		index = df.index[:dx.shape[0]]
	)

	shape = (len(sym), N)
	allocs = np.random.normal(size=shape)**2
	allocs = allocs / np.sum(allocs, axis=0)

	print(allocs.shape)

	portfolio =  (df.values @ allocs)
	# print(portfolio.shape)
	# print(df.values.shape)
	
	variances = [
		utils.volAdjust (
			basketVariance ( df.values.T, x, isLog = False )
		)
		for x in allocs.T
	]
	mu = portfolio.mean(axis=0)
	# print(mu.shape)
	# print(mu)
	mu = (1 + mu)**252 - 1

	priceDf = pd.DataFrame (
		data = allocs.T,
		columns = sym
	)
	priceDf['var'] = variances
	priceDf['mu'] = mu
	
	priceDf['sharpe'] = (priceDf['mu'] - riskFree) / priceDf['var'] **0.5
	
	priceDf.sort_values(by=['sharpe'], ascending=False, inplace=True)
	return priceDf
