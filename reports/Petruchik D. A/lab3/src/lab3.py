import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [6, 6],
    [-6, 6],
    [6, -6],
    [-6, -6]
], dtype=float)

E = np.array([0, 0, 1, 0], dtype=float)

T_MSE = np.where(E == 0, -1, 1)

alpha = 0.01
epochs = 500
Ee = 0.01

def net(x, w, b):
    return np.dot(w, x) + b


def step(u):
    return 1 if u >= 0 else 0


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def bce_loss(y, yhat):
    yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
    return -(y*np.log(yhat) + (1-y)*np.log(1-yhat))


def train_mse_fixed(X, T):

    w = np.zeros(2)
    b = 0
    history = []

    for epoch in range(epochs):

        Es = 0

        for x,t in zip(X,T):

            y = net(x,w,b)

            err = t - y

            Es += err**2

            w += alpha * err * x
            b += alpha * err

        history.append(Es)

        if Es <= Ee:
            break

    return w,b,history

def train_mse_adaptive(X,T):

    w = np.zeros(2)
    b = 0
    history = []

    for epoch in range(epochs):

        Es = 0

        for x,t in zip(X,T):

            y = net(x,w,b)

            err = t - y

            Es += err**2

            alpha_t = 1/(np.dot(x,x)+1)

            w += alpha_t * err * x
            b += alpha_t * err

        history.append(Es)

        if Es <= Ee:
            break

    return w,b,history

def train_bce_fixed(X,E):

    w = np.zeros(2)
    b = 0
    history = []

    for epoch in range(epochs):

        Es = 0

        for x,e in zip(X,E):

            z = net(x,w,b)

            yhat = sigmoid(z)

            err = yhat - e

            Es += bce_loss(e,yhat)

            w -= alpha * err * x
            b -= alpha * err

        history.append(Es)

        if Es <= Ee:
            break

    return w,b,history

def train_bce_adaptive(X,E):

    w = np.zeros(2)
    b = 0
    history = []

    for epoch in range(epochs):

        Es = 0

        for x,e in zip(X,E):

            z = net(x,w,b)

            yhat = sigmoid(z)

            err = yhat - e

            Es += bce_loss(e,yhat)

            alpha_t = 1/(np.dot(x,x)+1)

            w -= alpha_t * err * x
            b -= alpha_t * err

        history.append(Es)

        if Es <= Ee:
            break

    return w,b,history

w1,b1,h1 = train_mse_fixed(X,T_MSE)
w2,b2,h2 = train_mse_adaptive(X,T_MSE)
w3,b3,h3 = train_bce_fixed(X,E)
w4,b4,h4 = train_bce_adaptive(X,E)

plt.figure(figsize=(9,5))

plt.plot(h1,label="MSE fixed")
plt.plot(h2,label="MSE adaptive")
plt.plot(h3,label="BCE fixed")
plt.plot(h4,label="BCE adaptive")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Es")
plt.title("Сравнение сходимости")
plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(8,6))

for i in range(len(X)):

    if E[i]==0:
        plt.scatter(X[i,0],X[i,1],color="blue",s=120,label="class 0" if i==0 else "")
    else:
        plt.scatter(X[i,0],X[i,1],color="red",s=120,label="class 1")

x_line=np.linspace(-10,10,100)

y_mse=-(w1[0]*x_line+b1)/w1[1]
y_bce=-(w4[0]*x_line+b4)/w4[1]

plt.plot(x_line,y_mse,'--',label="MSE line")
plt.plot(x_line,y_bce,label="BCE line")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Разделяющие линии")
plt.legend()
plt.grid()

plt.show()

print("\nРежим тестирования")
print("Введите x1 x2 или q")

while True:

    s=input(">> ")

    if s=="q":
        break

    try:

        x1, x2 = map(float, s.replace(",", " ").split())

        z=net(np.array([x1,x2]),w4,b4)

        prob=sigmoid(z)

        cls=1 if prob>=0.5 else 0

        print("Вероятность =",round(prob,4))
        print("Класс =",cls)

    except:
        print("Ошибка ввода")