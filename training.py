from noggin import create_plot
from mynn.optimizers.sgd import SGD
import matplotlib.pyplot as plt
import mygrad as mg
import numpy as np
import model_setup as ms

# Setup
model = ms.Model('input size','hidden layer size',2) # TODO insert parameters
x_train, y_train, x_test, y_test = data() # TODO replace with correct data-retrieving function
optim = SGD(model.parameters, learning_rate=0.1)
batch_size = 25 # change based on total number of images in dataset

# plotter, fig, ax = create_plot(metrics=["loss", "accuracy"]) ### TODO uncomment when working in jupyter notebook

for epoch_cnt in range(10):
    idxs = np.arange(len(x_train))
    np.random.shuffle(idxs)  
    
    for batch_cnt in range(0, len(x_train) // batch_size):
        start_ind = batch_cnt*batch_size
        batch_indices = idxs[start_ind : start_ind+batch_size]
        batch = x_train[batch_indices]  # random batch of our training data

        pred = model(batch)
        pred_true = y_train[batch_indices]

        loss = softmax_cross_entropy(pred, pred_true)
        acc = ms.accuracy(pred, pred_true)

        loss.backward()
        optim.step()
        loss.null_gradients()

        # TODO uncomment in jupyter notebook
        # plotter.set_train_batch({"loss" : loss.item(),
        #                          "accuracy" : acc},
        #                          batch_size=batch_size)


    # TODO uncomment in jupyter notebook
    # test_idxs = np.arange(len(x_test))
    
    # for batch_cnt in range(0, len(x_test)//batch_size):
    #     start_ind = batch_cnt*batch_size
    #     batch_indices = test_idxs[start_ind : start_ind+batch_size]
        
    #     batch = x_test[batch_indices]
        
    #     pred_test = model(batch)
    #     true_test = y_test[batch_indices]

    #     test_accuracy = ms.accuracy(pred_test, true_test)        
    #     plotter.set_test_batch({"accuracy" : test_accuracy}, batch_size=batch_size)
    # plotter.set_test_epoch()

# evaluate model
pred = model(x_test)
accuracy = ms.accuracy(pred, y_test)
print("Accuracy: " + str(accuracy))