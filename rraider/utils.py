
def volAdjust ( vol, nPeriods = 252, alpha=2 ):
	"""Readjust volatility from 1 unit to nPeriods units.

	"""
	return vol * nPeriods **(1/alpha)
