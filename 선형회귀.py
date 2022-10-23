import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
print("scikit-learn version :",sklearn.__version__)

#Linear Regression
#공부시간에 따르 시험점수
dataset = pd.read_csv('LinearRegressionData.csv')
print(dataset.head())

X = dataset.iloc[:, :-1].values #처음부터 마지막 컬럼 직전까지의 데이터(독립변수)
y = dataset.iloc[:, -1].values #마지막 컬럼 데이터(종속변수-결과)
print(X,y)


reg = LinearRegression() #객체생성
reg.fit(X,y) #학습(모델생성)
y_pred = reg.predict(X) # X에 대한 예측값
print(y_pred) 

plt.scatter(X,y,color='blue') #산점도
plt.plot(X,y_pred,color='green') #선 그래프
plt.title('Score by hours') #제목
plt.xlabel('hours') #x축 이름
plt.ylabel('score') #y축 이름
plt.show()

print('9시간 공부했을 때 예상 점수 :',reg.predict([[9]]))
print(reg.coef_) #기울기 m
print(reg.intercept_) # y절편 b
