import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def generate_weights(tup = (3, 1)):
    """ generates x * y * n weights from -1 to 1 """
    return 2 * np.random.random(tup) - 1


def train(inputs, outputs, weights=None, iterations=1000):
    
    weights = generate_weights((inputs.shape[1], 1))

    for interations in range(iterations):
        fun = lambda x: sigmoid(np.dot(x, weights))
        out = fun(inputs) 
        error = outputs - out
        adjustments = error * sigmoid_derivative(out)
        weights += np.dot(inputs.T, adjustments)

    return fun 


if __name__ == '__main__':
    training_inputs = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])

    training_outputs = np.array([
        [0, 1, 1, 0]
    ]).T

    perceptron = train(training_inputs, training_outputs)
    
    x = np.array([1, 0 , 0])
    print(np.round(perceptron(x)))
