import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import math

class TorchModel(nn.Module):
    def __init__(self, input_dim, f1, f2, d1, num_classes):
        super(TorchModel, self).__init__()
        gain = math.sqrt(2.0)

        self.conv1 = nn.Conv2d(input_dim, f1, (5, 5), bias=True)
        nn.init.xavier_uniform_(self.conv1.weight, gain=gain)

        self.conv2 = nn.Conv2d(f1, f2, (5, 5), bias=True)
        nn.init.xavier_uniform_(self.conv2.weight, gain=gain)

        self.dense1 = nn.Linear(f2 * 37 * 37, d1, bias=True)
        nn.init.xavier_uniform_(self.dense1.weight, gain=gain)

        self.dense2 = nn.Linear(d1, num_classes, bias=True)
        nn.init.xavier_uniform_(self.dense2.weight, gain=gain)

    def __call__(self, x):
        relu = nn.ReLU()
        max_pool = nn.MaxPool2d((2, 2), 2)

        
        x = relu(self.conv1(x))
        x = max_pool(x)
        x = func.dropout(relu(self.conv2(x)), p=0.1)
        x = max_pool(x)
        x = relu(self.dense1(x.reshape(x.shape[0], -1)))
        return self.dense2(x)

    # @property
    # def parameters(self):
    #     params = []
    #     for layer in (self.conv1, self.conv2, self.dense1, self.dense2):
    #         params += list((layer.weight, layer.bias))
    #     return params

def accuracy(predictions, truth):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    return np.mean(np.argmax(predictions, axis=1) == truth)

def convert_data(images):
    images = images[:, np.newaxis, :, :]
        
    images = images.astype(np.float32)
    images /= 255.

    divide = 4 * len(images) // 5
    return (images[:divide], images[divide:])

# Saving and Loading
# Save: torch.save(model.state_dict(), PATH)
# Load: model.load_state_dict(torch.load(PATH))

