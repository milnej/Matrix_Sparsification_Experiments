from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import math as m

success_rate = np.random.rand(10,10)

cmap = colors.ListedColormap(['red', 'green'])
bounds = [0,.5,1]
norm = colors.BoundaryNorm(bounds,cmap.N)

fig,ax = plt.subplots()
ax.imshow(success_rate,cmap=cmap, norm=norm)

ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(-.5, 10, 1))
ax.set_yticks(np.arange(-.5, 10, 1))
x = np.linspace(0,10)
print(x)
plt.plot(x, np.sin(x))

plt.show()