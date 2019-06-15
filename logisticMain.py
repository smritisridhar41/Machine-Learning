import pandas as pandu
import numpy as numpu
import matplotlib.pyplot as plt


def sigmoid(x):
    res = 1/(1+numpu.exp(-1*x))
    return res


def plot_decision_boundary(x_matrix, y_matrix, theta):
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(x_matrix[:, 1], x_matrix[:, 2], c=y_matrix, marker='o', cmap='viridis_r')
    kasthi_theta = [-theta[0][0]/theta[2][0], -theta[1][0]/theta[2][0]]
    plt.plot(x_matrix[:, 1], numpu.dot(x_matrix[:, 0:2], kasthi_theta))
    plt.show()


def normalize_feature(x):
    x_norm = numpu.matrix.copy(x)
    mean_x = numpu.mean(x_norm,0)
    std_x = numpu.std(x_norm,0)
    for i in range(1,len(x_norm[0])):
        x_norm[:,i] = (x[:,i] - mean_x[i])/std_x[i]
    return x_norm


def gradient_descent(x_matrix,y_matrix,theta, alpha, num_of_iter):
    m = len(x_matrix)
    temp_theta = numpu.matrix.copy(theta)
    for i in range(num_of_iter):
        hypo_matrix = sigmoid(numpu.dot(x_matrix, theta))
        for j in range(len(theta)):
            theta[j][0] = temp_theta[j][0] - alpha / m * numpu.sum(
                numpu.multiply(numpu.subtract(hypo_matrix, y_matrix), numpu.row_stack(x_matrix[:, j])))
        temp_theta = numpu.matrix.copy(theta)
        # plot_decision_boundary(x_matrix,y_matrix,theta)
    return theta


def predict(theta):
    mark1, mark2 = input('Enter Two Marks : ').split()
    input_matrix = numpu.column_stack([1, float(mark1), float(mark2)])
    z = numpu.dot(input_matrix, theta)
    hypo_output = sigmoid(z)
    if hypo_output == 1:
        print('Admitted')
    else:
        print('Not Admitted')


def run():
    names = ['mark1', 'mark2', 'res']
    dataset = pandu.read_csv('ex2data1.txt', names=names)
    array = dataset.values

    # initialize values
    x = array[:, 0:2]
    x_matrix = numpu.column_stack([numpu.ones(len(x)), x])
    y = array[:, 2]
    y_matrix = numpu.row_stack(y)
    n = len(x_matrix[0])
    theta = numpu.zeros(shape=(n, 1))

    x_norm = normalize_feature(x_matrix)
    # real_theta = gradient_descent(x_matrix, y_matrix, theta, 3, 1500)
    # plot_decision_boundary(x_matrix, y_matrix, real_theta)
    real_theta = gradient_descent(x_norm,y_matrix,theta, 3, 50)
    plot_decision_boundary(x_norm, y_matrix, real_theta)
    print(real_theta)


    predict(real_theta)

if __name__ == '__main__':
    run()
