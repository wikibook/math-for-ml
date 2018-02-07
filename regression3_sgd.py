import numpy as np
import matplotlib.pyplot as plt

# 학습 데이터를 읽는다
train = np.loadtxt('click.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

# 표준화
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

# 매개변수를 초기화한다
theta = np.random.rand(3)

# 학습 데이터 행렬을 만든다
def to_matrix(x):
    return np.vstack([np.ones(x.size), x, x ** 2]).T

X = to_matrix(train_z)

# 예측함수
def f(x):
    return np.dot(x, theta)

# 평균제곱오차
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)

# 학습률
ETA = 1e-3

# 오차의 차분
diff = 1

# 갱신 횟수
count = 0

# 학습을 반복한다
error = MSE(X, train_y)
while diff > 1e-2:
    # 확률 경사하강법을 통해 매개변수를 갱신한다
    p = np.random.permutation(X.shape[0])
    for x, y in zip(X[p,:], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x

    # 이전 회에서 생긴 오차와의 차분을 계산한다
    current_error = MSE(X, train_y)
    diff = error - current_error
    error = current_error

    # 로그를 출력한다
    count += 1
    log = '{}회째: theta = {}, 차분 = {:.4f}'
    print(log.format(count, theta, diff))

# 그래프로 나타낸다
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()
