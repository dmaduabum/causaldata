# causaldata
**Easy data simulation for causal inference researchers**
`causaldata` is a Python package that makes it simple to generate realistic correlated data with mixed types (continuous, binary, ordinal, count) for power analysis, testing estimators, and debugging code.
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
```bash
pip install causaldata
```
Or install from source:
```bash
git clone https://github.com/dmaduabum/causaldata.git
cd causaldata
pip install -e .
```
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
## Documentation
Full documentation available at: [link coming soon]
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
## Contributing
Contributions welcome! Please open an issue or submit a pull request.

