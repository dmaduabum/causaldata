# causaldata/tests/test_mixed_simulator.py

import pytest
import numpy as np
import pandas as pd
from causaldata import MixedSimulator

def test_basic_generation():
    """Test basic data generation"""
    sim = MixedSimulator(n=100)
    sim.add_continuous("x1", mean=0, std=1)
    sim.add_binary("x2", prob=0.5)
    
    data = sim.generate(seed=42)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert list(data.columns) == ["x1", "x2"]

def test_continuous_bounds():
    """Test continuous variable bounds"""
    sim = MixedSimulator(n=1000)
    sim.add_continuous("x", mean=50, std=10, min_val=0, max_val=100)
    
    data = sim.generate(seed=42)
    
    assert data["x"].min() >= 0
    assert data["x"].max() <= 100

def test_correlations():
    """Test that correlations are approximately preserved"""
    sim = MixedSimulator(n=10000)
    sim.add_continuous("x1", mean=0, std=1)
    sim.add_continuous("x2", mean=0, std=1)
    sim.set_correlation("x1", "x2", 0.8)
    
    data = sim.generate(seed=42)
    
    observed_corr = data[["x1", "x2"]].corr().iloc[0, 1]
    assert abs(observed_corr - 0.8) < 0.05  # Within 5%

def test_ordinal_generation():
    """Test ordinal variable generation"""
    sim = MixedSimulator(n=1000)
    sim.add_ordinal("edu", levels=["HS", "College", "Grad"], probs=[0.3, 0.5, 0.2])
    
    data = sim.generate(seed=42)
    
    counts = data["edu"].value_counts(normalize=True)
    assert abs(counts["HS"] - 0.3) < 0.05
    assert abs(counts["College"] - 0.5) < 0.05
    assert abs(counts["Grad"] - 0.2) < 0.05

def test_reproducibility():
    """Test that same seed gives same results"""
    sim1 = MixedSimulator(n=100)
    sim1.add_continuous("x", mean=0, std=1)
    
    sim2 = MixedSimulator(n=100)
    sim2.add_continuous("x", mean=0, std=1)
    
    data1 = sim1.generate(seed=42)
    data2 = sim2.generate(seed=42)
    
    pd.testing.assert_frame_equal(data1, data2)
