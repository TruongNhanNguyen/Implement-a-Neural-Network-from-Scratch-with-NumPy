# Implement a Neural Network from Scratch with NumPy

![Implement a Neural Network from scratch with NumPy](asset/images/Impement%20NN%20from%20scratch%20with%20NumPy.png)

The best way to really understand how a neural network works is to implement one from scratch. That is exactly what I am going to do in this project. I will create a neural network class and design it in such a way to be more flexible. I don't wish to hardcode in it a specific activation of loss functions, or optimizer (that is SGD, Adam Boost or other Gradient-based mothods). The neural network will be designed to receive from outside the class so thar one can just take the class's code and pass to it whatever activation/loss/optimizer class that they wish to use hear as seperate things from `Neural Network` class. And we need both activation/loss functions and their derivatives.

To allow batch sizes greater than 1, the activation and loss functions should handle matrix input. Rows in these matrices will represent different data points, and the columns will be features. The network will allow for 2 kinds of activation functions: for hidden layers and the output layer. The hidden layer activations should operate on their input vectors elementwise, and thus their derivatives will also be elementwise, returning one vector for each data point. But the output activation should allow for each element in the output vector to be computed based on all the elements in the input vector. That is to be able to use softmax activation. Because of this, their derivatives needs to return a *Jacobian* matrix (a matrix consisting of the partial derivatives of each output function w.r.t. each of the input component; you can read more on Wikipedia) for each data point.

Here we will use only ReLU as hidden activation; identity and softmax will be used as output activations.

We used the `EPS` variable, which is the smallest positive representable number of `float64` type, to avoid division by 0. To avoid overflow errors in the `softmax` function, we subtracted the maximum of each data point from the input. We are allowed to do that because it does not change the output of the function as it has the same effect as dividing both terms of that fraction by the same amount.

The loss functions should take as input 2 matrices: the *predicted y* and *true y*, both of them of the same form as in the activation functions. These loss functions should output a single number for each data point. Their derivatives should output a row-vector for each data point, all of them stacked into an array of dimension 3. This output shape is required to be able to use NumPy’s `matmul()` function to multiply with the derivative of output activation. Note the use of `expand_dims()` function which is used to return the required shape.

Here we will use only stochastic gradient descent with momentum as an optimization method, but there are more gradient-based methods out there. Some popular choices are *Adam*, *RMSprop*, *Adagrad*. To allow the neural network class to work with all of these, we will implement the optimizer as a separate class with a `.update(old_params, gradient)` method that returns the updated parameters. The neural network class will receive an optimizer as a parameter. So, someone who wants to use other optimization methods can create a class with the required interface and pass it to the neural network class when instantiating.

To convert class labels in classification tasks to one-hot encoding we will use the `to_categorical()` utility function.

Now, let us start with the code of the NeuralNetwork class. The instantiation method expects the following parameters

- `layers`: a list consisting of the number of nodes in each layer (including input and output layers) e.g.: [5, 10, 2] means 5 inputs, 10 nodes in the hidden layer, and 2 output nodes
- `hidden_activation`: activation of hidden layers; a tuple of form (activation_function, its_derivative) This activation function and its derivative should perform their task elementwise on the input array e.g.: (`relu`, `d_relu`)
- `output_activation`: activation of output layer; a tuple of form (activation_function, its_derivative) This activation function takes as input an array of shape `(n, m)`; `n` samples, `m` neurons in the output layer; and returns an array of shape `(n, m)`; each element on a row in the output array is the output of a function of all the elements on that row in the input array. Its derivative takes as input an array similar to the one taken by the activation, but it returns an array of shape `(n, m, m)` which is a stack of Jacobian matrices, one for each sample. 
- `loss`: a tuple of form (loss_function, its_derivative) The loss function takes as input two arrays (*predicted y* and *true y*) of shape `(n, m)`; `n` samples, `m` neurons in output layer; and returns an array of shape `(n, 1)`, whose elements are the loss for each sample. Its derivative takes as input an array of shape `(n, m)` and returns one of shape `(n, 1, m)` which is a stack of row-vectors consisting of the derivatives w.r.t. each one of the m input variables. e.g.: (categorical_crossentropy, d_categorical_crossentropy) 
- `optimizer`: an object with a method `.update(old_params, gradient)` that returns the new params e.g.: `SGD()`

Then, it initializes its weights and biases using a variant of the *Xavier* initialization method. That is, we draw the weights and biases from a normal distribution with mean 0 and standard deviation of

$$\sigma = \sqrt{\dfrac{1}{\text{fan\_int} + \text{fan\_out}}}$$

where `fan_in` and `fan_out` are the number of nodes in the previous layer, respectively the number of neurons in the next layer. The number of rows in weights matrices matches the number of nodes in the previous layer, and the number of columns matches the number of nodes in the next layer. The biases are row vectors with the number of elements matching the number of nodes in the next layer.

To easily do the parameters update procedure we will create a `.__flatten_params(weights, biases)` method that transforms the list of weight matrices, and bias vectors received as input, to a flattened vector. We will also need a `.__restore_params(params)` method that turns a flattened vector of parameters back into lists of weights and biases. Note that the 2 underscores in front of the method name just means that the method is private in OOP terms. This just means that the method should be used only from inside the class.

The `.__forward(x)` method passes the input array x through the network and while it does so, it keeps track of the input and output arrays to and from each layer. Then it returns this as a list of which the i-th element is a list of the form [input to layer i, the output of layer i]. We will need those arrays to compute the derivatives in the backward pass.

The `.__backward(io_arrays, y_true)` method computes the gradient. It takes as input a list of the form returned by the `.__forward(x)` method and an array with the ground truth y. It computes the gradient of weights and biases using the backpropagation algorithm. Then it returns a tuple (d_weights, d_biases).

The method that orchestrates all the training is `.fit(x, y, batch_size, epochs, categorical)`, where:

- `x` is the input data
- `y` is the ground truth
- `batch_size` is the size of a batch of data
- `epochs` is the number of iterations through all the input data
- `categorical` is an optional parameter that, when set to true will convert `y` to one-hot encoding

For each batch of data, it uses `.__forward()` and `.__backward()` methods to compute the gradient then flatten the current parameters of the network and the gradient using `.__flatten_params()`. After that, computes the new parameters using `self.optimizer.update()`, then restores that returned vector to the right format with `.__restore_params()` and assigns that to `self.weights`, `self.biases`. At the end of each batch, the progress and average loss are printed. A list of all the loss values at the end of each epoch is maintained and returned.

By default, the `.predict()` method will return the exact values that are in the output nodes after the input x is passed through the network. If the labels parameter is set to true, then the predicted labels are returned; this is probably what you want in a classification problem.

The `.score()` method returns by default the average loss. If accuracy is set to true, then the accuracy will be returned. Note that in a classification problem, if you want the loss then y should be provided in one-hot encoding format, otherwise, if you want accuracy to be returned then y should be regular class labels.

Finally, we want to be able to save the parameters locally, so that we do not have to train our models each time we want to make a prediction. Note that the below methods can save and load just the weights and biases, but not the whole information about the layers, activations, loss function, and optimizer. So, you should also save the code used to instantiate the neural network.