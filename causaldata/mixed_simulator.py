# causaldata/causaldata/mixed_simulator.py

import numpy as np
import pandas as pd
from scipy.stats import norm

class MixedSimulator:
    """
    Simulate correlated mixed-type variables using Gaussian Copula.
    
    Examples
    --------
    >>> from causaldata import MixedSimulator
    >>> sim = MixedSimulator(n=1000)
    >>> sim.add_continuous("income", mean=50000, std=15000)
    >>> sim.add_binary("treated", prob=0.3)
    >>> sim.set_correlation("income", "treated", 0.5)
    >>> data = sim.generate(seed=42)
    """
    
    def __init__(self, n=1000):
        """
        Initialize the simulator.
        
        Parameters
        ----------
        n : int
            Number of observations to generate
        """
        self.n = int(n)
        self.variables = []
        self.var_names = []
        self.var_index = {}
        self.corr = None
    
    def add_continuous(self, name, mean=0, std=1, min_val=None, max_val=None):
        """
        Add a continuous variable with Normal(mean, std).
        
        Parameters
        ----------
        name : str
            Variable name
        mean : float
            Mean of the normal distribution
        std : float
            Standard deviation
        min_val : float, optional
            Minimum value (clips below)
        max_val : float, optional
            Maximum value (clips above)
        """
        spec = {
            "type": "continuous",
            "mean": float(mean),
            "std": float(std),
            "min_val": float(min_val) if min_val is not None else None,
            "max_val": float(max_val) if max_val is not None else None,
        }
        self._register(name, spec)
        return self
    
    def add_binary(self, name, prob=0.5):
        """
        Add a Bernoulli(p) variable.
        
        Parameters
        ----------
        name : str
            Variable name
        prob : float
            Probability of 1 (must be between 0 and 1)
        """
        if not (0 < prob < 1):
            raise ValueError("Binary prob must be in (0, 1).")
        spec = {
            "type": "binary",
            "prob": float(prob),
        }
        self._register(name, spec)
        return self
    
    def add_ordinal(self, name, levels, probs):
        """
        Add an ordinal/categorical variable.
        
        Parameters
        ----------
        name : str
            Variable name
        levels : list
            Category labels (e.g., ['HS', 'College', 'Grad'])
        probs : list
            Probabilities for each level (must sum to 1)
        """
        levels = list(levels)
        probs = np.array(probs, dtype=float)
        
        if np.abs(probs.sum() - 1) > 1e-8:
            raise ValueError("Ordinal variable 'probs' must sum to 1.")
        
        cutpoints = np.cumsum(probs)[:-1]
        
        spec = {
            "type": "ordinal",
            "levels": levels,
            "cutpoints": cutpoints,
        }
        self._register(name, spec)
        return self
    
    def _register(self, name, spec):
        """Internal method to register a new variable."""
        if name in self.var_names:
            raise ValueError(f"Variable '{name}' already exists.")
        
        idx = len(self.variables)
        self.var_index[name] = idx
        self.var_names.append(name)
        self.variables.append(spec)
        self._expand_corr_matrix()
    
    def _expand_corr_matrix(self):
        """Expand correlation matrix to match number of variables."""
        k = len(self.variables)
        if self.corr is None:
            self.corr = np.eye(k)
        else:
            new = np.eye(k)
            new[:-1, :-1] = self.corr
            self.corr = new
    
    def set_correlation(self, var1, var2, rho):
        """
        Set correlation between two latent variables.
        
        Parameters
        ----------
        var1 : str
            Name of first variable
        var2 : str
            Name of second variable
        rho : float
            Correlation coefficient (must be between -1 and 1)
        """
        if not (-1 < rho < 1):
            raise ValueError("Correlation must be in (-1, 1).")
        
        i = self.var_index[var1]
        j = self.var_index[var2]
        
        self.corr[i, j] = rho
        self.corr[j, i] = rho
        return self
    
    def generate(self, seed=None):
        """
        Generate a DataFrame with all simulated variables.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        df : pd.DataFrame
            Generated data with all specified variables
        """
        if seed is not None:
            np.random.seed(seed)
        
        k = len(self.variables)
        if k == 0:
            raise ValueError("No variables added. Add variables before generating.")
        
        # Step 1: Generate correlated standard normals
        Z = np.random.multivariate_normal(
            mean=np.zeros(k),
            cov=self.corr,
            size=self.n
        )
        
        # Step 2: Transform to uniform [0,1]
        U = norm.cdf(Z)
        
        # Step 3: Transform to target distributions
        out = {}
        
        for idx, spec in enumerate(self.variables):
            u_j = U[:, idx]
            vtype = spec["type"]
            name = self.var_names[idx]
            
            if vtype == "continuous":
                mu, sd = spec["mean"], spec["std"]
                values = mu + sd * norm.ppf(u_j)
                
                # Apply bounds
                min_val = spec.get("min_val")
                max_val = spec.get("max_val")
                
                if min_val is not None:
                    values = np.maximum(values, min_val)
                if max_val is not None:
                    values = np.minimum(values, max_val)
                
                out[name] = values
            
            elif vtype == "binary":
                p = spec["prob"]
                out[name] = (u_j < p).astype(int)
            
            elif vtype == "ordinal":
                cut = spec["cutpoints"]
                labels = np.array(spec["levels"])
                indices = np.digitize(u_j, cut)
                out[name] = labels[indices]
            
            else:
                raise NotImplementedError(f"Unknown type: {vtype}")
        
        return pd.DataFrame(out)
    
    def summary(self):
        """Print a summary of the simulator configuration."""
        print(f"MixedSimulator with {self.n} observations")
        print(f"\nVariables ({len(self.variables)}):")
        for name, spec in zip(self.var_names, self.variables):
            vtype = spec['type']
            if vtype == 'continuous':
                print(f"  {name}: Normal(μ={spec['mean']}, σ={spec['std']})", end="")
                if spec['min_val'] is not None or spec['max_val'] is not None:
                    print(f" [min={spec['min_val']}, max={spec['max_val']}]")
                else:
                    print()
            elif vtype == 'binary':
                print(f"  {name}: Binary(p={spec['prob']})")
            elif vtype == 'ordinal':
                print(f"  {name}: Ordinal{spec['levels']}")
        
        print(f"\nCorrelation Matrix:")
        corr_df = pd.DataFrame(
            self.corr, 
            index=self.var_names, 
            columns=self.var_names
        )
        print(corr_df.round(2))
