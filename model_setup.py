from mynn.initializers.glorot_uniform import glorot_uniform
from mynn.activations.relu import relu
from mynn.layers.conv import conv
from mygrad.nnet.layers import max_pool
from mygrad.nnet.losses import softmax_crossentropy
from mynn.layers.dense import dense
import numpy as np
import mygrad as mg


class Model:
    def __init__(self, input_dim, f1, f2, d1, num_classes):
        '''
        Parameters
        ----------
        input_dim:
            size of input based on training data
        n1 : int
            The number of neurons in the first hidden layer
        num_classes : int
            The number of classes predicted by the model'''
        init_kwargs = {'gain': np.sqrt(2)}
        self.conv1 = conv(input_dim, f1, 5, 5,
                          weight_initializer=glorot_uniform,
                          weight_kwargs=init_kwargs)
        self.conv2 = conv(f1, f2, 5, 5,
                          weight_initializer=glorot_uniform,
                          weight_kwargs=init_kwargs)
        self.dense1 = dense(f2 * 37 * 37, d1,
                            weight_initializer=glorot_uniform,
                            weight_kwargs=init_kwargs)
        self.dense2 = dense(d1, num_classes,
                            weight_initializer=glorot_uniform,
                            weight_kwargs=init_kwargs)

    def __call__(self, x):
        """ Performs a "forward-pass" of data through the network.

        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(M, ?)
            A batch of data consisting of M pieces of data,
            each with a dimentionality of ? (the number of
            values among all the pixels in a given image).

        Returns
        -------
        mygrad.Tensor, shape-(M, num_class)
            The model's prediction for each of the M images in the batch,
        """
        x = relu(self.conv1(x))
        x = max_pool(x, (2, 2), 2)
        x = relu(self.conv2(x))
        x = max_pool(x, (2, 2), 2)
        x = relu(self.dense1(x.reshape(x.shape[0], -1)))
        return self.dense2(x)

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.

        Returns
        -------
        List[mygrad.Tensor]
            A list of all of the model's trainable parameters
        """
        params = []
        for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
            params += list(layer.parameters)
        return params

    def save_model(self, path):
        """Path to .npz file where model parameters will be saved."""
        with open(path, "wb") as f:
            np.savez(f, *(x.data for x in self.parameters))

    def load_model(self, path):
        with open(path, "rb") as f:
            for param, (name, array) in zip(self.parameters, np.load(f).items()):
                param.data[:] = array


# accuracy function!!
def accuracy(predictions, truth):
    """
    Returns the mean classification accuracy for a batch of predictions.

    Parameters
    ----------
    predictions : Union[numpy.ndarray, mg.Tensor], shape=(M, D)
        The scores for D classes, for a batch of M data points

    truth : numpy.ndarray, shape=(M,)
        The true labels for each datum in the batch: each label is an
        integer in [0, D)

    Returns
    -------
    float
    """
    if isinstance(predictions, mg.Tensor):
        predictions = predictions.data
    return np.mean(np.argmax(predictions, axis=1) == truth)


def convert_data(images):
    '''
    returns xtrain,xtest as a tuple

    Parameters:
    ----------
    images: numpy array of image vectors

    Returns:
    --------
    A tuple containing the normalized images into (xtrain,xtest)
    '''
    images = images[:,np.newaxis,:,:]

    images = images.astype(np.float32)
    images /= 255.

    divide = 4 * len(images) // 5
    return (images[:divide], images[divide:])


