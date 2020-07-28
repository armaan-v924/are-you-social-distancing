from noggin import create_plot
from mynn.optimizers.sgd import SGD
import matplotlib.pyplot as plt
import mygrad as mg
import numpy as np
import model_setup as ms
from mynn.losses.cross_entropy import softmax_cross_entropy

# Setup
with_mask = np.load("with_masks.npy")
without_mask = np.load("without_masks.npy")

model = ms.Model(f1=20, f2=10, d1=20, input_dim=1, num_classes=2) # TODO insert parameters
x_train_mask, x_test_mask = ms.convert_data(with_mask) # TODO replace with correct data-retrieving function
x_train_without, x_test_without = ms.convert_data(without_mask) # TODO replace with correct data-retrieving function

y_train_mask = np.ones(x_train_mask.shape[0], dtype=np.int)
y_test_mask = np.ones(x_test_mask.shape[0], dtype=np.int)
y_train_without = np.zeros(x_train_without.shape[0], dtype=np.int)
y_test_without = np.zeros(x_test_without.shape[0], dtype=np.int)

x_train = np.append(x_train_mask, x_train_without, axis=0)
x_test = np.append(x_test_mask, x_test_without, axis=0)
y_train = np.append(y_train_mask, y_train_without, axis=0)
y_test = np.append(y_test_mask, y_test_without, axis=0)

optim = SGD(model.parameters, learning_rate=0.01, momentum=0.9, weight_decay=5e-04)
batch_size = 5 # change based on total number of images in dataset

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