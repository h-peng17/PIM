
import matplotlib 
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
import numpy as np 

loss = np.load("../data/loss.npy")
x = list(range(len(loss)))
loss = loss.tolist()
plt.plot(x, loss)
plt.savefig('../data/loss.png')