import numpy as np
import matplotlib.pyplot as plt
from hilbert import *
from functions import *
from ladder import *

rho = Density(Grid(-5, 5, 101))

x = rho.x.values
rho.values = np.exp(-(x - 1) ** 2) ** 2 + 0.5 * np.exp(-(x + 1) ** 2) ** 2
rho.particles = 5

plot_ladder_result(rho)
