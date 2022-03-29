import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8]
y = [1,2,3,4,5,6,7,8]

fig, ([raw, K1plt], [k3plt, k5plt]) = plt.subplots(2,2)
fig.suptitle('Vertically stacked subplots')
raw.plot(x, y)
K1plt.plot(x, y)
k3plt.plot(x, y)
k5plt.plot(x, y)

plt.show()