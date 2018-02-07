import numpy as np
import matplotlib.pyplot as plt

# 학습 데이터를 읽어들인다
train = np.loadtxt('images2.csv', delimiter=',', skiprows=1)
train_x = train[:,0:2]
train_y = train[:,2]

# 매개변수를 초기화한다
theta = np.random.rand(3)

# 표준화한다
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

# x0를 추가한다
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])

X = to_matrix(train_z)

# 시그모이드 함수
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

# 분류함수
def classify(x):
    return (f(x) >= 0.5).astype(np.int)

# 학습률
ETA = 1e-3

# 반복할 횟수
epoch = 5000

# 갱신할 횟수
count = 0

# 학습을 반복한다
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

    # 로그를 출력한다
    count += 1
    print('{}회째: theta = {}'.format(count, theta))

# 그래프로 나타내서 확인한다
x0 = np.linspace(-2, 2, 100)
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle='dashed')
plt.show()
