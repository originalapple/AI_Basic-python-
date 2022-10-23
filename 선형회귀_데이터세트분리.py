import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
print("scikit-learn version :",sklearn.__version__)

## 데이터 세트분리

dataset = pd.read_csv('LinearRegressionData.csv')
#print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0) 
#훈련 80 : 테스트 20으로 분리
# print(X),print(len(X)) #전체 데이터 X, 개수
# print(X_train), print(len(X_train)) # 훈련 세트 X_train
# print(X_test), print(len(X_test)) # 훈련 세트 X_test
# print(y),print(len(y)) # 전체 데이터 y
# print(y_train), print(len(y_train)) # 훈련 세트 y_train
# print(y_test), print(len(y_test)) # 훈련 세트 y_test

##분리된 데이터를 통한 모델링
reg = LinearRegression()
reg.fit(X_train, y_train) #훈련세트로 학습

## 데이터 시각화
plt.scatter(X_train, y_train,color='blue') #산점도
plt.plot(X_train, reg.predict(X_train),color='green') #선 그래프
plt.title('Score by hours (train data)') #제목
plt.xlabel('hours') #x축 이름
plt.ylabel('score') #y축 이름
plt.show()
## 데이터 시각화(테스트 세트)
plt.scatter(X_test, y_test,color='blue') #산점도
plt.plot(X_train, reg.predict(X_train),color='green') #선 그래프
plt.title('Score by hours (test data)') #제목
plt.xlabel('hours') #x축 이름
plt.ylabel('score') #y축 이름
plt.show()

print(reg.coef_) #기울기
print(reg.intercept_) #y절편

##모델 평가
print(reg.score(X_test, y_test)) #테스트 세트를 통한 모델평가 0~1사이의 값
print(reg.score(X_train, y_train)) #훈련 세트를 통한 모델평가 0~1사이의 값
