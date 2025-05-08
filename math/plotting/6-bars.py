#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']


fig, ax = plt.subplots()

ax.bar([0, 1, 2], fruit[0], color=colors[0], width=0.5, label="Apples")
ax.bar([0, 1, 2], fruit[1], color=colors[1], width=0.5, bottom=fruit[0], label="Bananas")
ax.bar([0, 1, 2], fruit[2], color=colors[2], width=0.5, bottom=fruit[0] + fruit[1], label="Oranges")
ax.bar([0, 1, 2], fruit[3], color=colors[3], width=0.5, bottom=fruit[0] + fruit[1] + fruit[2], label="Peaches")


ax.set_xlabel("Person")
ax.set_ylabel("Quantity of Fruit")
ax.set_title("Number of Fruit per Person")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Farrah', 'Fred', 'Felicia'])
ax.set_yticks(np.arange(0, 81, 10))
ax.legend()

plt.show()
