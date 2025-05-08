#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
x = np.arange(0, 11)

plt.plot(x, y, 'r-')  # Red line
plt.xlabel('x')
plt.ylabel('y = x^3')
plt.title('Line Graph')
plt.xlim(0, 10)  

plt.show()
