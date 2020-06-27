# This example of a MSDCNN
# For the actual experiments, a different implementation was used.
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import filters
import matplotlib as plt

class MyNN(nn.Module):
    def __init__(self, num_layers=5):
        super().__init__()
        for index in range(num_layers):
            # 1 + index input to 1 output
            # 3 by 3 kernel size
            # padding equal to dil to retain size
            dil = 1 + index % 10
            new_layer = nn.Conv2d(1 + index, 1, 3, stride=1, padding=dil, dilation = dil)
            self.add_module('L' + str(index), new_layer)

    def forward(self, x):
        for m in self.named_children():
            layer = m[1]
            y = layer(x)
            y = F.relu(y)
            # We concatenate the channels for each batch
            x = torch.cat((x,y),dim=1)
        # Intermediate images only needed for computation.
        return x[:,-1:,:]

class Learner():
    def __init__(self, data, solutions, loss = None):
        self.model = MyNN()
        # Data (N, c, imx, imy)
        # Data and solution must always have matching batch sizes
        # For MyNN the shape must be the same.
        self.data = data
        self.solutions = solutions

        self.lr = 0.3
        if loss == None:
            self.loss = F.mse_loss
        else:
            self.loss = loss
    
    def fit(self,eps, b_size):
        data = self.data
        solutions = self.solutions
        loss_fn = self.loss
        lr = self.lr

        N = data.size()[0]
        for ep in range(eps):
            for b_index in range(N // b_size):
                # Get training batch
                s_index = b_index * b_size
                e_index = (b_index + 1) * b_size
                batch_data = data[s_index:e_index]
                batch_solutions = solutions[s_index:e_index]

                # Make prediction
                pred = self.model(batch_data)
                loss = loss_fn(pred, batch_solutions)
                
                # Gradient descent.
                loss.backward(retain_graph=True)
                with torch.no_grad():
                    for p in self.model.parameters():
                        p -= p.grad * lr
                    self.model.zero_grad()
                print("ep: " + str(ep) + "- loss: " + str(loss))

if __name__ == "__main__":
    # Test the network
    N = 1000
    num_layer = 5
    imx, imy = 156, 156
    x = torch.randn(N,1,imx,imy)
    y = torch.tensor(filters.gaussian(x))
    lrnr = Learner(x, y)
    lrnr.fit(4, 100)

