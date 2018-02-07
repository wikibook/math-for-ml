import numpy as np
import matplotlib.pyplot as plt

# 진짜 함수
def g(x):
    return 0.1 * (x ** 3 + x ** 2 + x)

# 진짜 함수에 노이즈를 첨가한 학습 데이터를 적당한 개수만큼 준비한다
train_x = np.linspace(-2, 2, 8)
train_y = g(train_x) + np.random.randn(train_x.size) * 0.05

# 표준화
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

# 학습 데이터 행렬을 만든다
def to_matrix(x):
    return np.vstack([
        np.ones(x.size),
        x,
        x ** 2,
        x ** 3,
        x ** 4,
        x ** 5,
        x ** 6,
        x ** 7,
        x ** 8,
        x ** 9,
        x ** 10
    ]).T

X = to_matrix(train_z)

# 매개변수를 초기화한다
theta = np.random.randn(X.shape[1])

# 예측함수
def f(x):
    return np.dot(x, theta)

# 목적함수
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 정칙화 정수
LAMBDA = 0.5

# 학습률
ETA = 1e-4

# 오차
diff = 1

# 정칙화를 적용하지 않고 학습을 반복한다
error = E(X, train_y)
while diff > 1e-6:
    theta = theta - ETA * (np.dot(f(X) - train_y, X))

    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

theta1 = theta

# 정칙화를 적용해서 학습을 반복한다
theta = np.random.randn(X.shape[1])
diff = 1
error = E(X, train_y)
while diff > 1e-6:
    reg_term = LAMBDA * np.hstack([0, theta[1:]])
    theta = theta - ETA * (np.dot(f(X) - train_y, X) + reg_term)

    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

theta2 = theta

# 그래프로 나타낸다
plt.plot(train_z, train_y, 'o')
z = standardize(np.linspace(-2, 2, 100))
theta = theta1 # 정칙화 하지 않았슴
plt.plot(z, f(to_matrix(z)), linestyle='dashed')
theta = theta2 # 정칙화 했슴
plt.plot(z, f(to_matrix(z)))
plt.show()
