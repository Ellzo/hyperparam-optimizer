# hyperparam-optimizer: A Package for Hyperparameter Optimization

**hyperparam-optimizer** is a Python package designed to simplify the process of hyperparameter optimization for machine learning models. Currently, the package supports **genetic optimization** for models built with PyTorch. This project is under active development, and more optimization methods will be added in the future.

## Features

- Genetic algorithm-based optimization for hyperparameters.
- Easily customizable for PyTorch models.
- Example notebooks to demonstrate usage.

## Files Overview

### 1. `genetic.py`
The core implementation of the genetic algorithm for hyperparameter optimization. It includes functionality for:
- Generating initial populations of hyperparameters.
- Evolving populations based on fitness scores.
- Mutations and crossovers to explore the hyperparameter space.

### 2. `example1.ipynb`
A Jupyter notebook demonstrating the usage of the genetic optimization package on a basic PyTorch model.

### 3. `example2.ipynb`
Another example showcasing more advanced hyperparameter optimization using the genetic algorithm.

### 4. `LICENSE`
The license file outlines the terms of use. This package is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hyperparam-optimizer.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Explore the examples: Use Jupyter Notebook to open and run the provided examples:
   ```bash
   jupyter notebook
   ```

## Dependencies:
* `torch` and `torchvision`
* `pandas`
* `numpy`
* `matplotlib`

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
You are free to share and adapt the material under the following terms:
* **Attribution:**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* **NonCommercial:**: You may not use the material for commercial purposes.

For more details, see the LICENSE file or visit https://creativecommons.org/licenses/by-nc/4.0/.

## Future Plans
* Support for additional optimization techniques like Bayesian Optimization or Hyperband Search.
* Integration with other deep learning frameworks (e.g., TensorFlow).
* Additional examples and tutorials.
