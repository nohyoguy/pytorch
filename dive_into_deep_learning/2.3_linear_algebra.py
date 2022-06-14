# [2.3 Linear Algebra]
from matplotlib.pyplot import axis
import torch

# {2.3.1 Scalars}
x = torch.tensor(3.0)
y = torch.tensor(2.0)
# x = y, x * y, x / y, x**y

# {2.3.2 Vectors}
x = torch.arange(9)
x[3]

# {2.3.2.1 Lengyh, Dimensinality, and Shape}
len(x)

# {2.3.3 Matrices}
A = torch.arange(20).reshape(4, -1)
A.T  # matrix's transpose

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B == B.T

# {2.3.4 Tensors}
X = torch.arange(24).reshape(2, 3, 4)

# {2.3.5 Basic Properties of Tensor Arithmetic}
A = torch.arange(20).reshape(5, -1)
B = A.clone()  # Assign a copy of 'A' to 'B' by allocating new menory

a = 2
X = torch.arange(24).reshape(2, 3, 4)
(a*X).shape

# {2.3.6 Reduction}
# tensor 축소
x = torch.arange(4, dtype=torch.float32)
x.sum()

A.shape  # tensor의 크기
A.sum()  # tensor의 총합

A_sum_axix0 = A.sum(axis=0)  # 0 차원으로 합
A_sum_axix1 = A.sum(axis=1)  # 1 차원으로 합

A.sum(axis=[0, 1])   # A.sum()과 동일
A.sum() / A.numel()  # A.mean()
# A.mean(axis=0)  # A.sum(axis=0) / A.shape[0]

# {2.3.6.1 Non-Reduction Sum}
sum_A = A.sum(axis=1, keepdim=True)  # keepdim이 차원을 유지
A / sum_A * 100  # 각 차원의 Tensor의 비중을 나타냄

# {2.3.7 Dot Products}
x = torch.arange(4, dtype=torch.float32)    # x = [0, 1, 2, 3]
y = torch.ones(4, dtype=torch.float32)      # y = [1, 1, 1, 1]
torch.dot(x, y)     # x, y 내적
torch.sum(x * y)    # x, y 내적

# {2.3.8 Matrix-Vector Products}
A = torch.arange(20).reshape(5, 4)
x = torch.arange(4)
torch.mv(A, x)

# {2.3.9 Matrix-Matrix Multiplication}
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # FloatTensor
B = torch.ones(4, 3)
torch.mm(A, B)

# {2.3.10 Norms}
# 백터의 길이
u = torch.tensor([3.0, -4.0])
torch.norm(u)   # vector norm
torch.abs(u).sum()
torch.norm(torch.ones((4, 9)))  # matrix norm
