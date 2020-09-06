import warnings
warnings.filterwarnings('ignore')	


def discount ( cashFlow, riskFree, time ):
	"""Discount a single cash flow to the present.
	"""
	return cashFlow * (1+riskFree) ** -time



def annuityPrice ( cashFlow, riskFree, nPayments, timeOffset=0 ):
	"""Returns the sum of the current values of all cash flows from annuity.

	Has some problems where time-> infinity, and where riskFree=0
	"""
	return np.where(
		riskFree == 0.0,
		cashFlow * nPayments,
		(cashFlow / riskFree) * np.where(
			nPayments == np.inf,
			(1+riskFree)**(-timeOffset),
			(1+riskFree)**(-nPayments-timeOffset) * ((1+riskFree)**nPayments - 1)
		)
	)



def bond ( par, coupon, riskFree, nPayments, timeOffset=0):
	"""Returns the sum of the current values of all cash flows from bond.
	"""

	return discount(par, riskFree, nPayments+timeOffset) \
		+ annuityPrice(coupon, riskFree, nPayments, timeOffset)
