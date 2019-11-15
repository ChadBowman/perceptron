from numpy import dot, random, array, mean, concatenate, ones
from sympy import lambdify, symbols, E


class Perceptron:
    '''A single perceptron for making predictions, bitches'''

    def __init__(self, *, activation, training_set, targets, tolerance=0.01):
        '''Initializes partial derivaties

        Keyword arguments:
        activation -- the activation function
        training_set -- nxm matrix of inputs with known solutions
        targets -- n vector representing solution to training_set
        tolerance -- lowest possible error before terminating training
                     default: 0.01
        '''
        self.activation = lambdify(activation.free_symbols, activation)
        self.training_set = training_set
        self.targets = targets
        self.tolerance = tolerance

        self.partials = {}
        for sym in activation.free_symbols:
            self.partials[sym] = lambdify(sym, activation.diff(sym))

    def __call__(self, inputs):
        '''Invokes the perceptron to make a prediction

        inputs -- n vector of inputs
        '''
        product = dot(self._bias(inputs), self.weights)
        return self.activation(product)

    def __iter__(self):
        '''Resets weights and returns itself'''
        n = len(self.training_set[0]) + 1  # add 1 for bias
        self.weights = random.random((n, 1)) - 1
        return self

    def __next__(self):
        '''Takes a step through the training iteration.
        Raises StopIteration when all errors are below tolerance
        '''
        guess = self(self.training_set)
        error = self.targets - guess

        if all(self.tolerance > error):
            raise StopIteration

        adjustments = []
        for partial in self.partials.values():
            adjustments.append(error * partial(guess))

        biased_set = self._bias(self.training_set)
        self.weights += dot(biased_set.T, sum(adjustments))

        return error

    def _bias(self, matrix):
        '''Concatenates a bias value to each row'''
        shape = 1 if len(matrix.shape) == 1 else (matrix.shape[0], 1)
        axis = len(matrix.shape) - 1
        return concatenate((matrix, ones(shape=shape)), axis=axis)

    def train(self):
        '''Fully trains perceptron and returns self'''
        for error in self:
            pass
        return self


if __name__ == '__main__':
    training_inputs = array([
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    training_outputs = array([
        [0, 1, 1, 0]
    ]).T

    x = symbols('x')
    sig = 1 / (1 + E**(-x))

    harbinger = Perceptron(
        activation=sig,
        training_set=training_inputs,
        targets=training_outputs,
    )

    harbinger.train()

    print(round(harbinger(array([0, 0, 0]))[0]))
