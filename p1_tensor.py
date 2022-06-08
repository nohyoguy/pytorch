from numpy import dtype
import torch

# 초기화 되지 않은 텐서
x = torch.empty(4, 2)

# 무작위로 최가화된 텐서
# (random을 기준으로 초기화)
x = torch.rand(4, 2)

# 데이터 타입(dtype)이 long이고, 0으로 채워진 텐서
x = torch.zeros(4, 2, dtype=torch.long)

# 사용자가 입력한 값으로 텐서 초기화
x = torch.tensor([3, 2.3])

# 2 X 4 크기, double 타입, 1로 채워진 텐서
x = x.new_ones(2, 4, dtype=torch.double)

# x와 같은 크기, float 타입, 무작위로 채워진 텐서
x = torch.randn_like(x, dtype=torch.float)

# 텐서의 크기 계산
x.size()

# FloatTensor
ft = torch.FloatTensor([1, 2, 3])
"""
print(ft)
print(ft.short())
print(ft.int())
print(ft.long())
"""

# IntTensor
ft = torch.IntTensor([1, 2, 3])
"""
print(ft.float()) # 32bit
print(ft.double()) # 64bit
print(ft.half()) # 16bit
"""

# CUDA Tensors
# 맥북에는 GPU가 없으므로 실행 안됨
x = torch.randn(1)
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.device('cpu')
print(device)
y = torch.ones_like(x, device=device)
print(y)
x = x.to(device)
print(x)
z = x + y
print(z)
print(z.to('cpu', torch.double))
"""

# 다차원 텐서 표현
# 0D Tensor(Scalar)
t0 = torch.tensor(0)
'''
print(t0.ndim)
print(t0.shape)
print(t0)
'''
# 1D Tensor(Vactor)
t1 = torch.tensor([1, 2, 3])
'''
print(t1.ndim)
print(t1.shape)
print(t1)
'''
# 2D Tensor(Matrix)
t2 = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# 3D Tensor(Cube)
t3 = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# 4D Tensor
t4 = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],

                   [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],

                   [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
