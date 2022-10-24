import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
print("scikit-learn version :",sklearn.__version__)
# Gradient Descent
# 학습률 (Learning rate) -> 0.001, 0.003, 0.01, 0.03, 0.1, 0.3 주로 사용
# 에포크(Epoch) -> 모든 데이터를 한 번씩 사용하는 과정
# Stochastic Gradient Descent (확률적 경사하강법)

## 경사하강법
dataset = pd.read_csv('LinearRegressionData.csv')
#print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0) 

from sklearn.linear_model import SGDRegressor # SGD : Stochastic Gradient Descent 확률적 경사하강법
sr = SGDRegressor(max_iter=1000, eta0=1e-4, random_state=0, )#verbose=1) 
# 에포크를 지정할 수 있다 max_iter= : 훈련 세트 반복 횟수(Epoch 횟수)
# eta0= : 학습률(learning rate) 0.001 = 1e-3 : 지수 표기법으로 사용가능
# random_state= : 
sr.fit(X_train, y_train)

## 데이터 시각화
plt.scatter(X_train, y_train,color='blue') #산점도
plt.plot(X_train, sr.predict(X_train),color='green') #선 그래프
plt.title('Score by hours (train data, SGD)') #제목
plt.xlabel('hours') #x축 이름
plt.ylabel('score') #y축 이름
plt.show()

print('기울기 : {}'.format(sr.coef_)),print('y절편 :',sr.intercept_)
print("값 평가(TEST) : {}".format(sr.score(X_test,y_test))) # 테스트 세트를 통한 모델 평가
print("값 평가(Train) : {}".format(sr.score(X_train,y_train))) # 훈련 세트를 통한 모델 평가
