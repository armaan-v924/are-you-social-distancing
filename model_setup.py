from mynn.initializers.he_normal import he_normal
from mynn.activations.relu import relu
from mynn.optimizers.sgd import SGD
from mynn.losses.cross_entropy import softmax_cross_entropy
from mynn.layers.dense import dense

class Model:
    def __init__(self, input_dim, n1, num_classes):
        '''
        Parameters
        ----------
        input_dim: 
            size of input based on training data
        n1 : int
            The number of neurons in the first hidden layer
        num_classes : int
            The number of classes predicted by the model'''
        self.dense1 = dense(input_dim, n1, weight_initializer=he_normal)
        self.dense2 = dense(n1, num_classes, weight_initializer=he_normal)
    def __call__(self,x):
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
        return self.dense2(relu(self.dense1(x)))
    
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        Returns
        -------
        List[mygrad.Tensor]
            A list of all of the model's trainable parameters 
        """
        return (self.dense1.parameters + self.dense2.parameters)

#accuracy function!!
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
    images = images.reshape(len(images), -1).astype(np.float32)
    mean_image = images.mean(axis=0)
    std_image = images.std(axis=0)

    images -= mean_image
    images /= std_image

    divide = 4*len(images)//5
    return (images[:divide],images[divide:])


    