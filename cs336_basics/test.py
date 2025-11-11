import torch
token = "AB"
print(token.encode('utf-8'))
print(list(token.encode('utf-8')))
print(type(list(token.encode('utf-8'))[0]))


print(torch.arange(1, 10).shape)
