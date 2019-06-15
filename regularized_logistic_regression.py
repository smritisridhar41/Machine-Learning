import pandas as pandu
import numpy as numpu
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(x):
    res = 1 / (1 + numpu.exp(-1 * x))
    return res


def feature_map(x1, x2):
    degree = 6
    X = numpu.ones(shape=(len(x1), 1))

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            new_col = numpu.multiply(x1 ** (i - j), x2 ** j)
            if new_col.ndim == 1:
                new_col = numpu.column_stack(new_col)
            X = numpu.append(X, new_col, 1)

    return X


def cost(theta, x, y, lambu):
    m, n = x.shape
    t1 = numpu.log(sigmoid(x.dot(theta)))
    t2 = numpu.log(sigmoid(x.dot(theta)))

    t = numpu.multiply(-y, t1) - numpu.multiply(1 - y, t2)
    reg = lambu / (2 * m) + numpu.sum(numpu.power(theta, 2))
    cost_j = -(numpu.sum(t) / m) + reg
    return cost_j


def step_gradient(theta, x_matrix, y_matrix, lambu):
    m = len(x_matrix)
    n = len(x_matrix[0])
    temp_theta = numpu.copy(theta)
    temp_theta = numpu.reshape(temp_theta, (n, 1))
    hypo_matrix = sigmoid(numpu.dot(x_matrix, temp_theta))
    grad = numpu.zeros(shape=(len(temp_theta), 1))
    grad[0][0] = 1 / m * numpu.sum(
        numpu.multiply(numpu.subtract(hypo_matrix, y_matrix), numpu.row_stack(x_matrix[:, 0])))

    for j in range(1, len(temp_theta)):
        grad[j][0] = 1 / m * numpu.sum(
            numpu.multiply(numpu.subtract(hypo_matrix, y_matrix), numpu.row_stack(x_matrix[:, j]))) + lambu / m * \
                                                                                                      temp_theta[j][0]

    return grad


def big_step_gradient(x_matrix, y_matrix, theta, alpha, num_of_iter, lambu):
    for i in range(num_of_iter):
        grad = step_gradient(theta, x_matrix, y_matrix, lambu)
        theta = numpu.subtract(theta, alpha * grad)
    return theta


def predict(theta, X):
    probability = sigmoid(numpu.dot(X, theta))
    return [1 if x >= 0.5 else 0 for x in probability]


def predict_single_value(theta):
    result1, result2 = input('Enter Two Lab Results : ').split()
    input_matrix = numpu.column_stack([1, float(result1), float(result2)])
    input_featured = feature_map(input_matrix[:, 1], input_matrix[:, 2])
    hypo_output = predict(theta.T, input_featured)

    result = ['Accepted' if out >= 0.5 else 'Rejected' for out in hypo_output]
    print(result)


def plot_data(x_matrix, y_matrix):
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(x_matrix[:, 1], x_matrix[:, 2], c=y_matrix, marker='o', cmap='viridis_r')
    # plt.show()


def plot_decision_boundary(x, y, theta):
    plot_data(x, y)
    u = numpu.row_stack(numpu.linspace(-1, 1.5, 50))
    v = numpu.row_stack(numpu.linspace(-1, 1.5, 50))
    z = numpu.zeros(shape=(len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = numpu.dot(feature_map(u[i, :], v[j, :]), theta)

    u = numpu.reshape(u, len(u))
    v = numpu.reshape(v, len(v))
    z = z.transpose()
    plt.contour(u, v, z, 74)
    plt.show()


def calculate_accuracy(real_theta, x_featured, y_matrix):
    predictions = predict(real_theta, x_featured)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y_matrix)]
    accuracy = (sum(map(int, correct)) / len(correct)) * 100
    print('accuracy = {0}%'.format(accuracy))


def run():
    names = ['mark1', 'mark2', 'res']
    dataset = pandu.read_csv('ex2data2.txt', names=names)
    array = dataset.values

    # initialize values
    x = array[:, 0:2]
    x_matrix = numpu.column_stack([numpu.ones(len(x)), x])
    y = array[:, 2]
    y_matrix = numpu.row_stack(y)

    # create more features
    x_featured = feature_map(numpu.row_stack(x_matrix[:, 1]), numpu.row_stack(x_matrix[:, 2]))

    n = len(x_featured[0])  # number of features
    theta = numpu.zeros(shape=(n, 1))
    num_of_iter = 1500
    lambda_val = 0.003
    alpha = 0.01

    # truncated Newton algorithm
    real_theta_tnc, n_func_eval, rc = opt.fmin_tnc(func=cost, x0=theta, fprime=step_gradient,
                                                   args=(x_featured, y_matrix, lambda_val))
    print(real_theta_tnc)

    # run gradient descent
    real_theta_grad_des = big_step_gradient(x_featured, y_matrix, theta, alpha, num_of_iter, lambda_val)
    print(real_theta_grad_des)

    real_theta = real_theta_tnc
    plot_decision_boundary(x_matrix, y_matrix, real_theta)
    calculate_accuracy(real_theta, x_featured, y_matrix)
    predict_single_value(real_theta)


if __name__ == '__main__':
    run()
