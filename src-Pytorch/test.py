import torch 

print(torch.__version__)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())

x = torch.rand(5, 3)
print(x)



