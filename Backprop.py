import numpy as np
import matplotlib.pyplot as plt

def forward_prop(X, W1, b1, W2, b2):
    Z = 1/(1 + np.exp(- X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    ret = expA / expA.sum(axis=1, keepdims=True)
    return ret, Z

def classification_rate(Y, P):
    return np.mean(Y==P)

def cost(T, P):
    return (T * np.log(P)).sum()

def derivative_w2(Y, T, Z):
    N, M = Z.shape
    N, K = Y.shape
    ret = np.zeros([M,K])

    #slow
    # for i in range(N):
    #     for j in range(M):
    #         for k in range(K):
    #             ret[j,k] += (T[i][k] - Y[i][k]) * Z[i,j]

    #removing j
    # for i in range(N):
    #     for k in range(K):
    #         ret[:,k] += (T[i][k] - Y[i][k]) * Z[i,:]

    ret = Z.T.dot((T-Y))
    return ret

def derivative_b2(Y, T):
    ret = (T-Y).sum(axis = 0)
    return ret

def derivative_w1(Y, T, W2, Z, X):
    M, K = W2.shape
    N, D = X.shape
    ret = np.zeros([D,M])
    # for i in range(N):
    #     for j in range(M):
    #         for k in range(K):
    #             for l in range(D):
    #                 ret[l, j] += (T[i][k] - Y[i][k]) * W2[j][k] * Z[i][j] * ( 1 - Z[i][j]) * X[i][l]

    # for i in range(N):
    #         for k in range(K):
    #             for l in range(D):
    #                 ret[l, :] += (T[i][k] - Y[i][k]) * W2[:][k] * Z[i,:] * ( 1 - Z[i,:]) * X[i][l]

    # dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    #      (T - Y).dot(W2.T) * Z * (1 - Z)
    # ret2 = X.T.dot(dZ)
    ret = X.T.dot((T - Y).dot(W2.T) * Z * (1 - Z))
    return ret;

def derivative_b1(Y, T, W2, Z):
    ret = ((T-Y).dot(W2.T)*Z*(1-Z)).sum(axis=0)
    return ret


def main():
    # Generate blob data
    NClass = 500
    K = 3               # no of classes
    D = 2               # no of input features
    M = 3               # no of hidden layers
    X1 = np.random.randn(NClass, D) + np.array([2, 2])
    X2 = np.random.randn(NClass, D) + np.array([-2, 2])
    X3 = np.random.randn(NClass, D) + np.array([0, -2])
    X = np.row_stack([X1, X2, X3])

    Y = np.array([0] * NClass + [1] * NClass + [2] * NClass)
    N = len(Y)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()

    #Form indicator matrix
    T = np.zeros([N, K])
    for i in range(N):
        T[i, Y[i]] = 1

    #Initialize random weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    costs = []
    learning_rate = 10e-7

    for epoch in range(1000000):
        #Forward propatate
        output, hidden = forward_prop(X, W1, b1, W2, b2)

        if epoch%100 == 0:
            #Print cost and classification rate
            c = cost(T, output)
            costs.append(c)
            P = np.argmax(output, axis = 1)
            crate = classification_rate(Y, P)
            print("Cost: ", c, "Classification Rate: ", crate)

        #Back Propatation
        W2 += learning_rate * derivative_w2(output, T, hidden)
        b2 += learning_rate * derivative_b2(output, T)
        W1 += learning_rate * derivative_w1(output, T, W2, hidden, X)
        b1 += learning_rate * derivative_b1(output, T, W2, hidden)

if __name__ == '__main__'
    main()

