import numpy as np


def volAdjust ( vol, nPeriods = 252, alpha=2 ):
	"""Readjust volatility from 1 unit to nPeriods units.

	"""
	return vol * nPeriods **(1/alpha)


def mass_broadcast(*args):
	'''Cast some list of vectors of shape(a_i, 1) or (a_i,) to the same uniform (l,1)
	'''
	l = 0
	xs = []
	for i, x in enumerate(args):
		xs.append( np.asarray(x).reshape(-1,1) )
		l = max(l, xs[i].shape[0])

	for i, x in enumerate(xs):
		xs[i] = np.broadcast_to(x, (l, 1))

	return xs
