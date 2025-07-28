from setuptools import setup, find_packages

setup(
    name='bacali',
    version='0.1.0',
    author='Markus Flicker',
    author_email='markus.flicker@silicon-austria.com',
    description='A library for Bayesian model calibration using Hamiltonian Markov Chain Monte Carlo.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://lnzgitlab-internal01.research.silicon-austria.com/flickerm/bacali.git',
    packages=find_packages(),
    install_requires=[
        "arviz",
        "matplotlib",
        "numpy",
        "pandas",
        "xarray",
        "pymc",
        "scikit-learn",
        "jax",
        "jaxlib",
        "numpyro",
        "pyDOE",
        "json5",
        "PyYAML",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)