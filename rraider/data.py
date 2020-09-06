import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from joblib import Memory

cachedir = '/tmp/joblib'
memory = Memory(cachedir, verbose=0)


def getTsy ( currentDate, **kwargs):
	url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldAll'
	r = requests.get(url)
	r.raise_for_status()
	html = r.text

	soup = BeautifulSoup(html, 'html.parser')
	tables = soup.find('table', {'class': 't-chart'})
	df = pd.read_html(str(tables))[0]
	df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
	df = df.set_index('Date')

	if kwargs.get('floatDates', False):
		def f (name, data):
			x = int(name.split(' ')[0]) / 12.0
			if 'yr' in name:
				x *= 12.0
			return x, data
		df = pd.DataFrame(dict(
			f(n, df[n]) for n in df.columns
		))

	return df




def getJGB ( currentDate, **kwargs ):
	url = {
		'hist': 'https://www.mof.go.jp/english/jgbs/reference/interest_rate/historical/jgbcme_all.csv',
		'curr': 'https://www.mof.go.jp/english/jgbs/reference/interest_rate/jgbcme.csv',
	}
	currDf = pd.read_csv( url['curr'], header=1 )
	histDf = pd.read_csv( url['hist'], header=1 )
	df = histDf.append(currDf)

	df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
	df = df.set_index('Date')
	df.columns = [ x.replace('Y', ' yr') for x in df.columns ]
	df.replace('-', np.nan, inplace=True)
	for col in df.columns:
		df[col] = pd.to_numeric(df[col], errors='ignore')

	if kwargs.get('floatDates', False):
		def f (name, data):
			x = int(name.split(' ')[0]) / 12.0
			if 'yr' in name:
				x *= 12.0
			return x, data
		df = pd.DataFrame(dict(
			f(n, df[n]) for n in df.columns
		))
	return df




fs = {
	'tsy': {
		'f': getTsy
	},

	'jgb': {
		'f': getJGB
	}

}


def get( keyword, cache=True, **kwargs ):
	"""Acquires some data source.

	Arguments:
		keyword:
			keyword that identifies the dataset to be returned. Could be {'tsy'}
		cache:
			whether to stash this dataset for future use.
			Cache invalidates each day, so results are alway from today.
	
	Returns:
		pandas datetime instance.
	
	Examples:
		
		.. code-block:: python
		
		   type( ttools.data.get('tsy') )
		   # pandas.core.frame.DataFrame

		   ttools.data.get('tsy', floatDates=True)
		   # This one will have column names as the number of years until maturity
	"""

	x = fs[keyword]
	today = datetime.datetime.now().strftime('%Y-%m-%d')
	f = x['f']
	if cache:
		f = memory.cache(f)
	return f( today, **kwargs)
