import numpy as np

EPS = np.finfo(np.float64).eps


def to_categorical(labels):
    n_classes = labels.max()+1
    y = np.zeros((labels.shape[0], n_classes))

    for i in range(labels.shape[0]):
        y[i, labels[i]] = 1

    return y

# Activation functions + derivatives


def identity(x):
    return x


def d_identity(x):
    return np.tile(np.identity(x.shape[1]), (x.shape[0], 1, 1))


def relu(x):
    return np.maximum(x, 0)


def d_relu(x):
    return np.vectorize(lambda v: 1 if v > 0 else 0)(x)


def softmax(x):
    x = x - x.max(axis=1).reshape((-1, 1))
    exp = np.exp(x)
    s = np.sum(exp, axis=1).reshape((-1, 1))
    return exp/(s+EPS)


def d_softmax(x):
    s = softmax(x)
    D = np.stack([np.diag(s[i, :]) for i in range(s.shape[0])], axis=0)
    comb = np.matmul(np.expand_dims(s, 2), np.expand_dims(s, 1))
    return D - comb

# Loss functions + derivatives


def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred-y_true)**2, axis=1).reshape((-1, 1))


def d_mean_squared_error(y_pred, y_true):
    return np.expand_dims((2/y_pred.shape[1])*(y_pred-y_true), 1)


def categorical_crossentropy(y_pred, y_true):
    return -np.log(np.sum(y_true*y_pred, axis=1)+EPS)


def d_categorical_crossentropy(y_pred, y_true):
    return np.expand_dims(-y_true/(y_pred+EPS), 1)

# Optimizers


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, old_params, gradient):
        if not hasattr(self, 'delta_params'):
            self.delta_params = np.zeros_like(old_params)

        self.delta_params = self.momentum*self.delta_params - self.learning_rate*gradient
        new_params = old_params + self.delta_params

        return new_params

# Our Neural Network class


class NeuralNetwork:
    def __init__(self, layers, hidden_activation, output_activation, loss, optimizer):
        '''
        # Parameters:
            - layers: a list consisting of the number of nodes in each layer (including input and output layers)
                    e.g.: [5, 10, 2] means 5 inputs, 10 nodes in hidden layer, and 2 output nodes
            - hidden_activation: activation of hidden layers; a tuple of form (activation_function, its_derivative)
                    This activation function and its derivative should perform their task element-wise on the input array
                    e.g.: (`relu`, `d_relu`)
            - output_activation: activation of output layer; a tuple of form (activation_function, its_derivative)
                    This activation function takes as input an array of shape (n, m); n samples, m neurons in output layer;
                    and returns an array of shape (n, m); each element on a row is the output of a function of all the elements on that row.
                    Its derivative takes as input an array similar to the one taken by the activation, but it returns an array of shape
                    (n, m, m) which is a stack of Jacobian matrices, one for each sample.
            - loss: a tuple of form (loss_function, its_derivative)
                    The loss function takes as input two arrays (predicted y and true y) of shape (n, m); n samples, m neurons in output layer;
                    and returns an array of shape (n, 1), whose elements are the loss for each sample.
                    Its derivative takes as input an array of shape (n, m) and returns one of shape (n, 1, m) which is
                    a stack of row-vectors consisting of the derivatives w.r.t. each one of the m input variable
                    e.g.: (categorical_crossentropy, d_categorical_crossentropy)
            - optimizer: an object with a method `.update(old_params, gradient)` that returns the new params
                    e.g.: `SGD()`
        '''
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer

        self.weights = []
        self.biases = []
        self.nlayers = len(layers)
        nrows = layers[0]
        for i in range(1, self.nlayers):
            ncols = layers[i]
            std_dev = np.sqrt(1/(nrows+ncols))  # Xavier initialization
            self.weights.append(np.random.normal(
                size=(nrows, ncols), scale=std_dev))
            self.biases.append(np.random.normal(
                size=(1, ncols), scale=std_dev))
            nrows = ncols

    def __flatten_params(self, weights, biases):
        params = []
        for W in weights:
            params.append(W.flatten())
        for b in biases:
            params.append(b.flatten())

        params = np.concatenate(params)

        return params

    def __restore_params(self, params):
        weights = []
        biases = []

        start = 0
        for i in range(1, self.nlayers):
            nrows = self.layers[i-1]
            ncols = self.layers[i]
            end = start+nrows*ncols
            p = params[start:end]
            W = p.reshape((nrows, ncols))
            weights.append(W)
            start = end

        for i in range(1, self.nlayers):
            ncols = self.layers[i]
            end = start+ncols
            p = params[start:end]
            b = p.reshape((1, ncols))
            biases.append(b)
            start = end

        return (weights, biases)

    def __forward(self, x):
        io_arrays = []
        for i in range(self.nlayers):
            if i > 0:
                x = np.matmul(x, self.weights[i-1]) + self.biases[i-1]
            layer_io = [x]  # input to layer i
            if i == self.nlayers-1:
                activation = self.output_activation[0]
            elif i > 0:
                activation = self.hidden_activation[0]
            else:
                def activation(v): return v
            x = activation(x)
            layer_io.append(x)  # output of layer i
            io_arrays.append(layer_io)
        return io_arrays

    def __backward(self, io_arrays, y_true):
        e = self.loss[1](io_arrays[-1][1], y_true)

        batch_size = y_true.shape[0]
        d_weights = []
        d_biases = []
        for i in range(self.nlayers-1, 0, -1):
            if i == self.nlayers-1:
                e = np.matmul(e, self.output_activation[1](io_arrays[i][0]))
                e = np.squeeze(e, 1)
            else:
                e = e * self.hidden_activation[1](io_arrays[i][0])
            d_w = np.matmul(io_arrays[i-1][1].transpose(), e)/batch_size
            d_b = np.mean(e, axis=0)
            d_weights.append(d_w)
            d_biases.append(d_b)
            e = np.matmul(e, self.weights[i-1].transpose())

        d_weights.reverse()
        d_biases.reverse()

        return (d_weights, d_biases)

    def fit(self, x, y, batch_size, epochs, categorical=False):
        if categorical:
            y = to_categorical(y)

        y_ncols = y.shape[1]

        n_samples = x.shape[0]

        epoch_loss = []
        for i in range(epochs):
            xy = np.concatenate([x, y], axis=1)
            np.random.shuffle(xy)
            x, y = np.split(xy, [-y_ncols], axis=1)

            print(f'Epoch {i+1}/{epochs}\n')
            start = 0
            loss_hist = []
            while start < n_samples:
                end = min(start+batch_size, n_samples)
                x_batch = x[start:end, :]
                y_batch = y[start:end, :]

                io_arrays = self.__forward(x_batch)
                d_weights, d_biases = self.__backward(io_arrays, y_batch)

                params = self.__flatten_params(self.weights, self.biases)
                gradient = np.nan_to_num(
                    self.__flatten_params(d_weights, d_biases))

                params = self.optimizer.update(params, gradient)

                self.weights, self.biases = self.__restore_params(params)

                loss_hist.append(
                    np.mean(self.loss[0](io_arrays[-1][1], y_batch)))
                print(f'{end}/{n_samples} ; loss={np.mean(loss_hist)}', end='\r')
                if end >= n_samples:
                    print('\n')
                start = end
            epoch_loss.append(np.mean(loss_hist))
        return np.array(epoch_loss)

    def predict(self, x, labels=False):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)

        output = self.__forward(x)[-1][1]

        if labels:
            return np.argmax(output, 1)
        else:
            return output

    def score(self, x, y, accuracy=False):
        if accuracy:
            return np.mean(self.predict(x, True) == y)
        else:
            output = self.predict(x)
            return np.mean(self.loss[0](output, y))

    def save_params(self, filename):
        np.save(filename, self.__flatten_params(self.weights, self.biases))

    def load_params(self, filename):
        params = np.load(filename)
        self.weights, self.biases = self.__restore_params(params)
