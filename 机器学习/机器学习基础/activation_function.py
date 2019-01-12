import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10)
y_sigmoid = 1/(1+np.exp(-x))
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

fig = plt.figure()
# plot sigmoid
ax = fig.add_subplot(221)
ax.plot(x,y_sigmoid)
ax.grid()
ax.set_title('(a) Sigmoid')

# plot tanh
ax = fig.add_subplot(222)
ax.plot(x,y_tanh)
ax.grid()
ax.set_title('(b) Tanh')

# plot relu
ax = fig.add_subplot(223)
y_relu = np.array([0*item  if item<0 else item for item in x ]) 
ax.plot(x,y_relu)
ax.grid()
ax.set_title('(c) ReLu')

#plot leaky relu
ax = fig.add_subplot(224)
y_relu = np.array([0.2*item  if item<0 else item for item in x ]) 
ax.plot(x,y_relu)
ax.grid()
ax.set_title('(d) Leaky ReLu')

plt.tight_layout()
plt.show()