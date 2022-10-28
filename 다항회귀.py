from turtle import color
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
from sklearn.preprocessing import PolynomialFeatures #다항회귀 때 쓰는 것
print("scikit-learn version :",sklearn.__version__)

## 03. Polynomial Regression 다항회귀
# 공부시간에 따른 시험점수(우등생버전)
dataset = pd.read_csv('PolynomialRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

## 3.1 단순선형회귀 (Simple Linear Regression)
reg = LinearRegression()
reg.fit(X, y) # 전체 데이터로 학습
# 데이터 시각화(전체)
plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, reg.predict(X), color='green') #선그래프
plt.title('Scored by hours (genius)') #제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
print('전체 데이터 평가 : {}'.format(reg.score(X, y)))

## 3.2 다항회귀(Polynomial Regression)
poly_reg = PolynomialFeatures(degree=4) # 4차 다항식
X_poly = poly_reg.fit_transform(X)
#X_poly[:5] # [x] -> [x^0, x^1, x^2] -> x가 3이라면 [1,3,9]으로 변환
# X[:5]
poly_reg.get_feature_names_out()
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) # 변환된 X와 y를 가지고 모델생성 (학습)

# 데이터 시각화(변환된 X와 y)
plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='green') #선그래프
plt.title('Scored by hours (genius)') #제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

X_range = np.arange(min(X), max(X), 0.1) # X의 최소값에서 최대값까지의 범위를 0.1단위로 생성
#X_range.shape
X_range = X_range.reshape(-1, 1) # row 개수는 자동으로 계산, column 개수는 1개
# 데이터 시각화(변환된 X와 y)
plt.scatter(X, y, color='blue') # 산점도
plt.plot(X_range, lin_reg.predict(poly_reg.fit_transform(X_range)), color='green') #선그래프
plt.title('Scored by hours (genius)') #제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

## 공부시간에 따른 시험 성적 예측
study_pred = reg.predict([[2]]) #2시간을 공부했을 때 선형회귀 모델의 예측
print(study_pred)
study_pred = lin_reg.predict(poly_reg.fit_transform([[2]])) #2시간을 공부했을 때 다항회귀 모델의 예측
print(study_pred)
#차수가 높아지면 과대적합이 발생할 수 있다 
print('데이터 평가 : {}'.format(lin_reg.score(X_poly, y)))
