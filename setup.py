import setuptools

setuptools.setup(
        name="ttools",
        version="0.1.0",
        author="Brett Graves",
        author_email="alienbrett648@gmail.com",
        description="Trading Tools",
        packages=setuptools.find_packages(),
		install_requires=[
			'joblib>=0.16.0',
			'numpy>=1.19.1',
			'pandas>=1.1.0',
			'pyally>=1.1.0',
			'scikit-learn>=0.23.2',
			'scipy>=1.5.2',
			'yfinance>=0.1.54'
		],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
        ],
)
