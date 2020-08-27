  
#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path
import os

working_dir = path.abspath(path.dirname(__file__))
ROOT = os.path.abspath(os.path.dirname(__file__))

# Read the README.
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

setup(name='bco',
      version='0.0.1',
      description='Bayesian Collaborative Optimization',
      long_description=README,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=['tests*']),
      setup_requires=["cython", "numpy", "torch", "torchvision"],
      install_requires=["numpy", "scipy", "matplotlib", "jupyter", "scikit-learn", "tqdm",
                        "pytest"],
      )