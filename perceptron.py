import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def generate_weights(tup = (3, 1)):
    return 2 * np.random.random(tup) - 1


def update_weights(weights, inputs, outputs, targets):
    error = targets - outputs
    adjustments = error * sigmoid_derivative(outputs)
    return weights + np.dot(inputs.T, adjustments)


def train(training_set, targets, weights=None, iterations=1000):
    
    weights = generate_weights((training_set.shape[1], 1))

    for interations in range(iterations):
        model = lambda x: sigmoid(np.dot(x, weights))
        result = model(training_set)
        weights = update_weights(weights, training_set, result, targets)

    return model 


if __name__ == '__main__':
    training_inputs = np.array([
        [0, 0, 1], # 0
        [1, 1, 1], # 1
        [1, 0, 1], # 1
        [0, 1, 1]  # 0
    ])

    training_outputs = np.array([
        [0, 1, 1, 0]
    ]).T

    perceptron = train(training_inputs, training_outputs)
    
    x = np.array([1, 0, 0])
    result = perceptron(x)[0]

    print(f'{x} : {np.round(result)}')
