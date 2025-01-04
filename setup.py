from setuptools import setup, find_packages

setup(
    name='shearnet',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'jax',
        'flax',
        'galsim',
        'ngmix',
        'numpy',
        'tqdm',
        'scipy',
        'optax',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'shearnet-train=shearnet.cli:main',
            'shearnet-eval=shearnet.evaluate:main',
        ],
    },
)

