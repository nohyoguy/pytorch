# [2.2 Data Preprocessing]
# 날것의 데이터를 전달하기전에 가공한 데이터를 텐서 포멧이 적용한다.
import pandas as pd
import torch
import os

# ../data/house_tiny.csv에 artifical dataset 생성
# csv 파일 생성 및 작성
# 빈 값은 NA로 처리
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# {2.2.1 Reading the Dataset}
# pandas package(pd)를 이용하여 csv 파일을 불러옴
data = pd.read_csv(data_file)

# {2.2.2 Handling Missing Data}
# intenger-location based indexing(iloc)
# fillna으로 결측값 처리 mean을 이용하여 NaN에 평균(mean)값으로 대체
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs.iloc[:, 0] = inputs.fillna(inputs.iloc[:, 0].mean())
inputs = pd.get_dummies(inputs, dummy_na=True)

print(inputs)
print(outputs)

# 2.2.3. Conversion to thr Tensor Format
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
