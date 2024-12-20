
<!-- ![image](relaxloss.jpg) -->
# Network Topology Protection Framework

This repository contains the implementation for ["SecureNT: A Practical Framework for Efficient Topology Protection and Monitoring"](https://arxiv.org/abs/2412.08177).

Contact: Chengze Du ([ducz0338@dupt.edu.cn](ducz0338@bupt.edu.cn))

## Structure

```
.
├── algorithm_infer_tp/      # Topology inference algorithms
│   ├── MLE.py              # Maximum Likelihood Estimation
│   └── MLTP.py             # MCMC-based Topology Learning
├── algorithm_obf_tp/        # Topology protection algorithms
│   ├── AntiTomo.py         # Anti-tomography protection
│   ├── Proto.py            # Prototype-based protection
│   └── SecureNT.py         # Secure Network Topology protection
├── Alg_NTP_evaluate.py     # Network Tomography link performance evaluate after Topology Protection
```

## Topology Inference Algorithms

### MLE (Maximum Likelihood Estimation)
- Implements topology inference using maximum likelihood estimation
- Uses Gurobi optimizer for solving the optimization problem


### MLTP (MCMC-based Topology Learning)
- Implements topology inference using Markov Chain Monte Carlo


## Topology Protection Algorithms
- AntiTomo
- Proto
- SecureNT


## Requirements

- Python 3.7+
- NumPy
- NetworkX
- Gurobi Optimizer
- Matplotlib
- SciPy

## Usage

Each algorithm can be used independently or as part of the complete framework. Example usage:

```python
# Topology Inference
from algorithm_infer_tp.MLE import MLE
from algorithm_infer_tp.MLTP import TopologyIdentification

# Topology Protection
from algorithm_obf_tp.AntiTomo import AntiTomoTopologyObfuscation
from algorithm_obf_tp.Proto import TopologyObfuscation
```

## License

[MIT License](LICENSE)
