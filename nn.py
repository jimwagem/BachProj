import torch
x = torch.rand(4, requires_grad=True)
A = torch.rand(4,4)
y = A @ x
v = torch.rand(4)

y.backward(v)
print(x.grad)

z = torch.dot(v,x)
z.backward()
print(x.grad)
