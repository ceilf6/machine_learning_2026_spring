# 1. Import from top-level package (most common)
from simple_nn_utseus import Dense, ReLU, load_mnist_from_csv

# 2. Import subpackages
from simple_nn_utseus.layers import Dense, ReLU
from simple_nn_utseus.data import load_mnist_from_csv

# 3. Import specific modules
from simple_nn_utseus.layers.dense import Dense
from simple_nn_utseus.layers.activation import ReLU

# 4. Import entire subpackage
import simple_nn_utseus.layers as layers

network = [layers.Dense(784, 64), layers.ReLU()]

# 5. What you should NOT do
from simple_nn_utseus.layers import _utils  # Not in __all__
from simple_nn_utseus.data import _preprocessing  # Not exported
