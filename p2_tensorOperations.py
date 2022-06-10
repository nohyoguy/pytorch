import torch
import math

a = torch.rand(3, 3)
# print(a)
a = a * 2 - 1
# print(a)
'''
print(torch.abs(a))  # 절댓값
print(torch.ceil(a))  # 올림
print(torch.floor(a))  # 내림
print(torch.clamp(a, -0.5, 0.5))  # 한정

print(torch.min(a)) # 최솟값
print(torch.max(a)) # 최댓값
print(torch.mean(a)) # 평균
print(torch.std(a)) # 표준편차
print(torch.prod(a)) # 곱
print(torch.unique(torch.tensor([1, 2, 3, 1, 2, 2]))) # 중목제외
'''

'''
x = torch.rand(3, 3, 3)
# argmax
print(x.max(dim=0))
print(x.max(dim=1))
print(x.max(dim=2))
# argmin
print(x.min(dim=0))
print(x.min(dim=1))
print(x.min(dim=2))
'''

# torch.add
x = torch.rand(2, 2)
y = torch.rand(2, 2)
torch.add(x, y)

result = torch.empty(2, 2)
torch.add(x, y, out=result)

y.add_(x)  # in-place (내부에서 변동 +=와 비슷)

# torch.sub
x = torch.rand(2, 2)
y = torch.rand(2, 2)
torch.sub(x, y)
x.sub(y)  # torch.sub(x, y)와 동일
x.sub_(y)  # in-place, 값이 변경됨

# torch.mul
x = torch.rand(2, 2)
y = torch.rand(2, 2)
# print(x * y)
torch.mul(x, y)
x.mul_(y)

# torch.div
x = torch.rand(2, 2)
y = torch.rand(2, 2)
# print(x / y)
torch.div(x, y)
x.div_(y)

# torch.mm (내적_dot product)
x = torch.rand(2, 2)
y = torch.rand(2, 2)
torch.matmul(x, y)
z = torch.mm(x, y)
torch.svd(z)
