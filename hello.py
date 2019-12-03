import numpy as np
import matplotlib.pyplot as plt

greyhounds, labs = 500, 500

greyHeight = 28 + 4 * np.random.randn(greyhounds)
labHeight = 28 + 4 * np.random.randn(labs)
plt.hist([greyHeight, labHeight], stacked=True, color=["r", "b"])
plt.show()
