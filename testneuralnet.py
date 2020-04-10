import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5,10)
        self.lin2 = nn.Linear(10,5)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


train_data = []
solutions = []
num_samples = 100
real_lin = nn.Linear(5,5)
for _ in range(num_samples):
    x = torch.randn(5)
    y = real_lin(torch.sin(x))
    train_data.append(x)
    solutions.append(y)

solutions = torch.stack(solutions)
train_data = torch.stack(train_data)
loss_func = F.mse_loss

model = MyNN()


lr = 0.1
def fit(eps):
    for ep in range(eps):
        pred = model(train_data)
        loss = loss_func(pred, solutions)

        loss.backward(retain_graph=True)
        with torch.no_grad():
            for p in model.parameters():
                p -= p.grad * lr
            model.zero_grad()

# print(loss_func(model(train_data), solutions))
# fit(20)
# print(loss_func(model(train_data), solutions))
