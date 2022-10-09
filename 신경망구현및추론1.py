import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
def softmax(x):
    max = np.max(x)
    exp_x = np.exp(x-max) #overflow를 방지
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x
    return y

## 1. Setting the weight of 2-layer neural network
w1 = np.array([[0.4,0.2,0.3],[0.1,0.2,0.1]])
b1 = np.array([0.1,0.2,0.2])

w2 = np.array([[0.1,0.4],[0.2,0.3],[0.4,0.6]])
b2 = np.array([0.2,0.1])

## 2. Perform inference for an input pattern
x = np.array([2.2,1.3]) #입력샘플

# 계층 1 연산
v1 = np.dot(x,w1) + b1
h1 = sigmoid(v1)
print("v1:",v1), print("h1:",h1)

#계층 2 연산
v2 = np.dot(h1,w2) + b2
h2 = sigmoid(v2)
print("v2:",v2), print("h2:",h2)

# 2.3 소프트맥스함수 적용
y = softmax(h2)

print("입력 :",x)
print("출력 :",y)
