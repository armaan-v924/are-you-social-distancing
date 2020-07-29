import torch
import torch_model as tm
import numpy as np
from noggin import create_plot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with_mask = np.load("with_masks.npy")
without_mask = np.load("without_masks.npy")


model = tm.TorchModel(f1=20, f2=10, d1=20, input_dim=1, num_classes=2).to(device)
x_train_mask, x_test_mask = tm.convert_data(with_mask)
x_train_without, x_test_without = tm.convert_data(without_mask)

y_train_mask = np.ones(x_train_mask.shape[0], dtype=np.int)
y_test_mask = np.ones(x_test_mask.shape[0], dtype=np.int)
y_train_without = np.zeros(x_train_without.shape[0], dtype=np.int)
y_test_without = np.zeros(x_test_without.shape[0], dtype=np.int)

x_train = np.append(x_train_mask, x_train_without, axis=0)
x_test = np.append(x_test_mask, x_test_without, axis=0)
y_train = np.append(y_train_mask, y_train_without, axis=0)
y_test = np.append(y_test_mask, y_test_without, axis=0)
optim = torch.optim.SGD(model.parameters, lr=0.01, momentum=0.9, weight_decay=5E-4) 

# plotter, fig, ax = create_plot(metrics=["loss", "accuracy"]) ### TODO uncomment when working in jupyter notebook

loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 30

for epoch_cnt in range(10):
    idxs = np.arange(len(x_train))
    np.random.shuffle(idxs)

    for batch_cnt in range(0, len(x_train) // batch_size):
        optim.zero_grad()
        start_ind = batch_cnt*batch_size
        batch_indices = idxs[start_ind : start_ind+batch_size]
        batch = x_train[batch_indices]  # random batch of our training data
        pred = model(torch.Tensor(batch).to(device))
        pred_true = y_train[batch_indices]

        loss = loss_fn(pred, torch.from_numpy(pred_true).long().to(device))
        loss.backward()
        acc = tm.accuracy(pred.cpu(), pred_true)
        optim.step()
        
        # TODO uncomment in jupyter notebook
        plotter.set_train_batch({"loss" : loss.item(),
                                 "accuracy" : acc},
                                batch_size = batch_size)
          
    # TODO uncomment in jupyter notebook
    # test_idxs = np.arange(len(x_test))
    
    # for batch_cnt in range(0, (len(x_test) // batch_size)):
    #     start_ind = batch_cnt*batch_size
    #     batch_indices = test_idxs[start_ind : start_ind+batch_size]
        
    #     batch = x_test[batch_indices]
        
    #     pred_test = model(torch.Tensor(batch).to(device))
    #     true_test = y_test[batch_indices]

    #     test_accuracy = tm.accuracy(pred_test.cpu(), true_test)        
    #     plotter.set_test_batch({"accuracy" : test_accuracy}, batch_size=batch_size)
    # plotter.set_test_epoch()

pred = model(torch.Tensor(x_test).to(device))
accuracy = tm.accuracy(pred.cpu(), y_test)
print("Accuracy: " + str(accuracy))