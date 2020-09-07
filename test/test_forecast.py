import unittest

from rraider.forecast.multi import *
from rraider.forecast.single import *

import pandas_datareader.data as web


class TestSingle(unittest.TestCase):
	
	def test_brownian_shape(self):
		'''Test the shape of brownian outputs
		'''
		# Create a model with starting val 100, mean 0.01, sigma 0.2, and 50 resolution
		m = BrownianModel( 100, 0.01, 0.2, 50 )
		x,y = m.simulate( 10, 99, path=True )

		self.assertEqual( x.shape, (500,) )
		self.assertEqual( y.shape, (99,500) )

		y = m.simulate ( 10, 99, path=False )
		self.assertEqual ( y.shape, (99,) )



	def test_logbrownian_shape(self):
		'''Test the shape of brownian outputs
		'''
		# Create a model with starting val 100, mean 0.01, sigma 0.2, and 50 resolution
		m = LogBrownianModel( 100, 0.01, 0.2, 50 )
		x,y = m.simulate( 10, 99, path=True )

		self.assertEqual( x.shape, (500,) )
		self.assertEqual( y.shape, (99,500) )

		y = m.simulate ( 10, 99, path=False )
		self.assertEqual ( y.shape, (99,) )



	def test_logbrownian_shape(self):
		'''Test the shape of brownian outputs
		'''

		import warnings
		warnings.filterwarnings('ignore')
		market = web.DataReader( 'SPY', 'yahoo' )

		# Create a model with starting val 100, mean 0.01, sigma 0.2, and 50 resolution
		m = GarchModel( market['Close'], p=3, q=3, mean='zero', dist='skewt' )
		x,y = m.simulate( 10, 99, path=True )

		self.assertEqual( x.shape, (10,) )
		self.assertEqual( y.shape, (99,10) )

		y = m.simulate ( 10, 99, path=False )
		self.assertEqual ( y.shape, (99,) )



class TestMulti(unittest.TestCase):
	
	def test_brownian_shape(self):
		'''Test the shape of brownian outputs
		'''
		# Create a model with starting val 100, mean 0.01, sigma 0.2, and 50 resolution
		m = MultiBrownianModel( 100, 0.01, [[1.0, -0.5],[-0.5,1.0]], 50 )
		x,y = m.simulate( 10, 99, path=True )

		self.assertEqual( x.shape, (500,) )
		self.assertEqual( y.shape, (99,500,2) )

		y = m.simulate ( 10, 99, path=False )
		self.assertEqual ( y.shape, (99,2) )

	def test_logbrownian_shape(self):
		'''Test the shape of brownian outputs
		'''
		# Create a model with starting val 100, mean 0.01, sigma 0.2, and 50 resolution
		m = MultiLogBrownianModel( 100, 0.01, [[1.0, -0.5],[-0.5,1.0]], 50 )
		x,y = m.simulate( 10, 99, path=True )

		self.assertEqual( x.shape, (500,) )
		self.assertEqual( y.shape, (99,500,2) )

		y = m.simulate ( 10, 99, path=False )
		self.assertEqual ( y.shape, (99,2) )

