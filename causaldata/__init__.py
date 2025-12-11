# causaldata/causaldata/__init__.py

"""
causaldata: Easy Data Simulation for Causal Inference
======================================================

A Python package for simulating realistic correlated data with mixed types
for power analysis, testing estimators, and debugging code in causal inference.

Main Classes
------------
MixedSimulator : Simulate correlated mixed-type variables using Gaussian copula

Example
-------
>>> from causaldata import MixedSimulator
>>> sim = MixedSimulator(n=1000)
>>> sim.add_continuous("income", mean=50000, std=15000)
>>> sim.add_binary("treated", prob=0.3)
>>> sim.set_correlation("income", "treated", 0.5)
>>> data = sim.generate(seed=42)
"""

__version__ = "0.1.0"

from .mixed_simulator import MixedSimulator

__all__ = ["MixedSimulator"]
