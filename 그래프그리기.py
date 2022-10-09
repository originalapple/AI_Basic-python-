import numpy as np
import matplotlib.pyplot as plt

#그래프1
plt.figure(figsize=(8,4))
x = [10,20,30,40]
y = [1,4,9,16]
plt.plot(x,y,"b-")
plt.plot(x,y,"r^")
plt.grid(True, ls="--",lw=1)
plt.show()

#그래프2
plt.figure(figsize=(8,4))
x = [10,20,30,40]
y = [1,4,9,16]
plt.plot(x,y,c="b",lw=2,ls=":",ms=15,marker="o",mew=5,mec="g",mfc="r")
plt.grid(True, ls="--",lw=1)
plt.show()
