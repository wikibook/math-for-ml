import numpy as np
import matplotlib.pyplot as plt

# 학습 데이터를 읽어들인다
train = np.loadtxt('images1.csv', delimiter=',', skiprows=1)
train_x = train[:,0:2]
train_y = train[:,2]

# 웨이트를 초기화한다
w = np.random.rand(2)

# 식별함수
def f(x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1

# 반복할 횟수
epoch = 10

# 갱신 횟수
count = 0

# 웨이트를 학습한다
for _ in range(epoch):
    for x, y in zip(train_x, train_y):
        if f(x) != y:
            w = w + y * x

    # 로그를 출력한다
    count += 1
    print('{}회째: w = {}'.format(count, w))

# 그래프로 나타내서 확인한다
x1 = np.arange(0, 500)
plt.plot(train_x[train_y ==  1, 0], train_x[train_y ==  1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.plot(x1, -w[0] / w[1] * x1, linestyle='dashed')
plt.show()
