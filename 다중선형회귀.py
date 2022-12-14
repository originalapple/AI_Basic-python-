import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("scikit-learn version :",sklearn.__version__)

## 2. Multiple Linear Regression 다중선형회귀
# 독립변수가 2개 이상
## One-Hot Encoding
## Multicollinearity 다중공선성 
# 독립변수들간에 서로 강한 상관관계를 가지면서 회귀계수 추정의 오류가 나타나는 문제
# 즉, 하나의 피처가 다른 피처에 영향을 미침
# Dummy Column이 n개이면 n-1개만 사용 -> Dummy variable trap

dataset = pd.read_csv('MultipleLinearRegressionData.csv')
X = dataset.iloc[:, :-1].values # 
y = dataset.iloc[:,-1].values # 마지막 열만 가져온다. 
#print(X)
#print(y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')
X = ct.fit_transform(X)
#print(X)
# 1 0 : Home
# 0 1 : Library
# 0 0 : Cafe
## 데이터 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
## 학습(다중 선형 회귀)
reg = LinearRegression()
reg.fit(X_train, y_train)
## 예측 값과 실제 값 비교(테스트 세트)
y_pred = reg.predict(X_test)
print('y 예측 값 :',y_pred)
print("y 테스트값 :", y_test)
# 모델 평가
print("훈련 세트 평가 : {}".format(reg.score(X_train,y_train)))
print("테스트 세트 평가 : {}".format(reg.score(X_test,y_test)))

## 회귀모델평가
## 다양한 평가 지표(회귀 모델)
# MAE (Mean Absolute Error) : (실제 값과 예측 값) 차이의 절대값
# MSE (Mean Squared Error) : 차이의 제곱
# RMSE (Root Mean Squared Error) : 차이의 제곱에 루트
# R2 : 결정 계수
# R2는 1에 가까울수록 좋다. 나머지는 0에 가까울수록 좋다

print("MAE 값 : {}".format(mean_absolute_error(y_test, y_pred))) #실제 값, 예측 값  MAE
print("MSE 값 : {}".format(mean_squared_error(y_test, y_pred))) #실제 값, 예측 값  MSE
print("RMSE 값 : {}".format(mean_squared_error(y_test, y_pred, squared=False))) #실제 값, 예측 값  RMSE
print("R2 값 : {}".format(r2_score(y_test, y_pred))) #R2