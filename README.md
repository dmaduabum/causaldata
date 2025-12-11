# causaldata

**Easy data simulation for causal inference researchers**

`causaldata` is a Python package that makes it simple to generate realistic correlated data with mixed types (continuous, binary, ordinal) for power analysis, testing estimators, and debugging code.

## Features

**Current (v0.1.0):**
- Generate correlated mixed-type variables using Gaussian copula
- Support for continuous, binary, and ordinal data
- Direct control over pairwise correlations
- Optional bounds for continuous variables

**Planned:**
- `PanelSimulator`: Repeated observations over time
- Missing data patterns (MAR, MCAR, MNAR)
- Conditional dependencies
- Parallel generation for large-scale power studies

## Installation

Install from source:
```bash
git clone https://github.com/dmaduabum/causaldata.git
cd causaldata
pip install -e .
```

**Requirements:**
- Python 3.8+
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## Quick Start
```python
from causaldata import MixedSimulator

# Initialize simulator
sim = MixedSimulator(n=1000)

# Add variables
sim.add_continuous("income", mean=50000, std=15000, min_val=0)
sim.add_binary("treated", prob=0.3)
sim.add_ordinal("education",
                levels=["HS", "College", "Grad"],
                probs=[0.3, 0.5, 0.2])

# Set correlations
sim.set_correlation("income", "education", 0.5)
sim.set_correlation("treated", "education", 0.2)

# Generate data
data = sim.generate(seed=42)
print(data.head())
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Basic usage
- Correlation visualization


## How It Works

`causaldata` uses a Gaussian copula approach:
1. Generate correlated standard normals
2. Transform to uniform [0,1] using Gaussian CDF
3. Transform each variable to its target distribution

This preserves correlation structure while allowing different marginal distributions.

## Testing

Run tests with:
```bash
pytest tests/
```

## Citation

If you use `causaldata` in your research, please cite:
```bibtex
@software{causaldata2025,
  author = {Maduabum, Dili},
  title = {causaldata: Easy Data Simulation for Causal Inference},
  year = {2025},
  url = {https://github.com/dmaduabum/causaldata}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Dili Maduabum - dilimaduabum@gmail.com

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

