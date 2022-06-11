# 2.1 Data Manipulation
import torch

# 0 ~ n-1 까지 출력
x = torch.arange(12, dtype=torch.float32)

x.shape
x.numel()    # tensor의 개수를 나타냄

X = x.reshape(3, 4)  # vector -> matrix

torch.zeros((2, 3, 4))  # tensor의 모든값 0
torch.ones((2, 3, 4))   # tensor의 모든값 1
torch.randn(3, 4)       # tensor 랜덤
torch.tensor([[2, 1, 4, 3],
              [1, 2, 3, 4],
              [4, 3, 2, 1]])   # tensor 직접 지정

# [2.1.2 Operations]
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# x+y, x-y, x*y, x/y, x**y
torch.exp(x)    # 지수함수

# concatenate (cat)
X = torch.arange(8, dtype=torch.int).reshape(2, 2, 2, 1) + 1
Y = torch.arange(8, dtype=torch.int).reshape(2, 2, 2, 1) * 10 + 10

torch.cat((X, Y), dim=0)
torch.cat((X, Y), dim=1)
torch.cat((X, Y), dim=2)
torch.cat((X, Y), dim=3)

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
X == Y

# [2.1.3 Broadcasting Mechanism]
a = torch.arange(3).reshape((3, 1))  # vector a : [0, 1, 2] (3 X 1)
b = torch.arange(2).reshape((1, 2))  # vector b : [0, 1]T (1 X 2)
a+b  # Matrix (3 X 2)


# [2.1.4 Indexing and Slicing]
X[-1]
X[1:3]
X[1, 2]  # X[1][2]와 같다.

X[0:2, :2] = 12

# [2.1.5 Savinc Memory]
Y = Y + X   # Y에 새로운 주소가 할당된다 (id(Y)가 달라진다)
Z = torch.zeros_like(Y)  # Y와 같은 크기에 값을 0으로 채운다.
Z[:] = Y + X    # Z에 기존 주소로 유지
X += Y  # X에 기존 주소로 유지 (X[:] = X + Y와 같음)

# [2.1.6 Conversion to Other Python Objects]
A = X.numpy()
B = torch.from_numpy(A)
type(A)  # <class 'numpy.ndarray'>
type(B)  # <class 'torch.Tensor'>

a = torch.tensor([3.5])
a.item()    # 스칼라 값만 입력
float(a)    # 스칼라 값만 입력
int(a)      # 스칼라 값만 입력
