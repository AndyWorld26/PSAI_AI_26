import numpy as np
import matplotlib.pyplot as plt

# обучающая выборка
X = np.array([
    [2, 1],
    [-2, 1],
    [2, -1],
    [-2, -1]
], dtype=float)

y = np.array([0, 1, 0, 0], dtype=float)

epochs = 5000
alpha = 0.01
eps = 0.01


# сигмоида
def sigmoid(z):
    return 1/(1+np.exp(-z))


# BCE
def bce(pred, y):
    e = 1e-9
    return -(y*np.log(pred+e)+(1-y)*np.log(1-pred+e))


# =========================
# MSE фиксированный шаг
# =========================
def mse_fixed(X,y):

    w=np.random.randn(2)
    b=np.random.randn()

    errors=[]

    for epoch in range(epochs):

        total=0

        for xi,yi in zip(X,y):

            z=np.dot(w,xi)+b
            pred=z

            err=yi-pred
            total+=err**2

            w+=alpha*err*xi
            b+=alpha*err

        errors.append(total/len(X))

        if errors[-1] <= eps:
            break

    return w,b,errors


# =========================
# MSE адаптивный шаг
# =========================
def mse_adaptive(X,y):

    w=np.random.randn(2)
    b=np.random.randn()

    errors=[]
    t=1

    for epoch in range(epochs):

        total=0

        for xi,yi in zip(X,y):

            lr=1/t
            t+=1

            z=np.dot(w,xi)+b
            pred=z

            err=yi-pred
            total+=err**2

            w+=lr*err*xi
            b+=lr*err

        errors.append(total/len(X))

        if errors[-1]<=eps:
            break

    return w,b,errors


# =========================
# BCE фиксированный шаг
# =========================
def bce_fixed(X,y):

    w=np.random.randn(2)
    b=np.random.randn()

    errors=[]

    for epoch in range(epochs):

        total=0

        for xi,yi in zip(X,y):

            z=np.dot(w,xi)+b
            pred=sigmoid(z)

            total+=bce(pred,yi)

            grad=(pred-yi)

            w-=alpha*grad*xi
            b-=alpha*grad

        errors.append(total/len(X))

        if errors[-1]<=eps:
            break

    return w,b,errors


# =========================
# BCE адаптивный шаг
# =========================
def bce_adaptive(X,y):

    w=np.random.randn(2)
    b=np.random.randn()

    errors=[]
    t=1

    for epoch in range(epochs):

        total=0

        for xi,yi in zip(X,y):

            lr=1/t
            t+=1

            z=np.dot(w,xi)+b
            pred=sigmoid(z)

            total+=bce(pred,yi)

            grad=(pred-yi)

            w-=lr*grad*xi
            b-=lr*grad

        errors.append(total/len(X))

        if errors[-1]<=eps:
            break

    return w,b,errors


# обучение моделей
w1,b1,e1=mse_fixed(X,y)
w2,b2,e2=mse_adaptive(X,y)
w3,b3,e3=bce_fixed(X,y)
w4,b4,e4=bce_adaptive(X,y)


# =========================
# график ошибок
# =========================
plt.figure(figsize=(8,5))

plt.plot(e1,label="MSE fixed")
plt.plot(e2,label="MSE adaptive")
plt.plot(e3,label="BCE fixed")
plt.plot(e4,label="BCE adaptive")

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Convergence comparison")

plt.legend()
plt.grid()

plt.show()


# =========================
# визуализация точек
# =========================
plt.figure(figsize=(7,7))

for i in range(len(X)):

    color="red" if y[i]==1 else "blue"
    plt.scatter(X[i,0],X[i,1],color=color,s=120,edgecolors="black")


x=np.linspace(-3,3,300)

# линия BCE adaptive
if w4[1]!=0:
    yline=-(w4[0]*x+b4)/w4[1]
    plt.plot(x,yline,label="BCE adaptive",linewidth=2)

# линия MSE fixed
if w1[1]!=0:
    yline2=-(w1[0]*x+b1)/w1[1]
    plt.plot(x,yline2,"--",label="MSE fixed",linewidth=2)

plt.xlim(-3,3)
plt.ylim(-3,3)

plt.grid()
plt.legend()

plt.title("Decision boundaries")
plt.xlabel("X1")
plt.ylabel("X2")

plt.show()


# =========================
# режим функционирования
# =========================
def predict(x1,x2):

    z=w4[0]*x1+w4[1]*x2+b4
    p=sigmoid(z)

    cls=1 if p>=0.5 else 0

    print("Probability:",round(p,4))
    print("Class:",cls)

    plt.figure(figsize=(7,7))

    for i in range(len(X)):
        color="red" if y[i]==1 else "blue"
        plt.scatter(X[i,0],X[i,1],color=color,s=120,edgecolors="black")

    if w4[1]!=0:
        yline=-(w4[0]*x+b4)/w4[1]
        plt.plot(x,yline)

    plt.scatter(x1,x2,color="magenta",s=200,marker="X")

    plt.xlim(-3,3)
    plt.ylim(-3,3)

    plt.grid()
    plt.show()


while True:

    v1=input("X1 (exit): ")

    if v1.lower()=="exit":
        break

    v2=input("X2: ")

    try:
        predict(float(v1),float(v2))
    except:
        print("Input error")