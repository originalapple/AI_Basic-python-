import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def relu(x):
    return np.maximum(0,x)

x = np.arange(-4.1,4.1,0.1)

plt.plot(x, sigmoid(x), '-', label="Logistic sigmoid")
plt.plot(x, tanh(x), '--', label="Hyperbolic tangent")
plt.plot(x, relu(x), '-', label="ReLU")

plt.plot(0, sigmoid(0),"ko",0,tanh(0),"ko") # x=0일 때의 함수값
plt.title("Activation functions")
plt.ylim(-1.5,2.1)

plt.grid(color="#BDBDBD", linestyle='-',linewidth=0.5)
plt.legend()
plt.show()