import torch

# 텐서의 조작 (Manipulation)
# 인덱싱(Indexing)
x = torch.tensor([[1, 2], [3, 4]])
'''
print(x)

print(x[0, 0])
print(x[0, 1])
print(x[1, 0])
print(x[1, 1])

print(x[:, 0])
print(x[:, 1])

print(x[0, :])
print(x[1, :])
'''

# view: 텐서의 크기나 모양을 변경
# 기본적으로 변경 전과 후에 텐서 안의 원소 개수가 유지
# -1로 설정되면 계산을 통해 해당 크기값을 유추
'''
x = torch.randn(4, 5)
y = x.view(20)
z = x.view(5, -1)
w = x.view(2, 5, -1)
'''

# item: 텐서에 값이 단 하나라도 존재하면 숫자값을 얻을 수 있음
# 스칼라값 하나만 사용해야한다!
'''
x = torch.randn(1)
print(x)
print(x.item())
print(x.dtype)
'''

# squeeze: 차원을 축소(제거)
'''
tensor = torch.rand(1, 3, 3)
print(tensor)
print(tensor.shape)

t = tensor.squeeze() # 차원 제거
print(t)
print(t.shape)
'''

# unsqueeze: 차원을 증가
'''
t = torch.rand(3, 3)
print(t)
print(t.shape)
tensor = t.unsqueeze(dim=2) # 선택한 차원에 대해 추가
print(tensor)
print(tensor.shape)
'''

# stack: 텐서간 결합
'''
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
print(torch.stack([x, y, z]).size())
'''

# cat(concatenate): 텐서를 결합하는 메소드
# stack과 유사하지만 쌓을 dim이 존재해야함
# 해당 차원을 늘려준 후 결합
'''
a = torch.randn(1, 3, 3)
print(a)
b = torch.rand(1, 3, 3)
print(b)
c = torch.cat((a, b), dim=2)
print(c)
print(c.size())
'''

a = torch.tensor([3, 3],
                 [4, 3]
                 [4, 3]
                 [4.5, 2]
                 [4.5, 3]
                 [0, 3])
