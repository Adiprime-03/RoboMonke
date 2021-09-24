import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

table = pd.read_csv('coords.csv')

x = np.array([table['x'][0], table['x'][0], table['x'][1]])
y = np.array([480 - table['y'][0], 480 - table['y'][1], 480 - table['y'][1]])

plt.xlim(0, 640)
plt.ylim(0, 480)

plt.plot(x, y)
plt.show()
