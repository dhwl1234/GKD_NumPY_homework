import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 存储年龄和身高数据
age_data = []
height_data = []

# 用csv库打开csv文件并读取数据
# with open('height.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         age_data.append(float(row[0]))
#         height_data.append(float(row[1]))

# 用pandas库读取文件
df = pd.read_csv('height.csv', usecols=[0, 1])
age_data = df.iloc[:, 0].values
height_data = df.iloc[:, 1].values

# 用hist统计男青少年年龄
# age_bins = [i * 2 for i in range(11)]
# plt.hist(age_data, bins=age_bins, color='blue')  # 用hist函数实现直方图统计
#
# 用np.bincount计算
age_bins = [i * 2 + 1 for i in range(10)]
age_hist = np.bincount(np.array(age_data).astype(int))
age_hist = np.cumsum(age_hist.reshape(-1, 2), axis=1)[:, -1]
plt.bar(age_bins, age_hist, width=2, color='blue')

plt.title('Age distribution of male teenagers')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.figure(2)
# 用hist统计男青少年身高
# height_bins = [i * 10 for i in range(10, 19)]
# plt.hist(height_data, bins=height_bins, color='green')


# # 用np.bincount计算
height_bins = [i * 10+5 for i in range(10, 18)]
height_hist = np.bincount(np.array(height_data).astype(int))
# 数据的拆解变形，计算每个统计点对应的数据
height_hist1 = height_hist[:int(height_hist.size / 10) * 10].reshape(int(height_hist.size / 10), 10)
height_hist2 = height_hist[int(height_hist.size / 10) * 10:]
height_hist2 = height_hist[int(height_hist.size / 10) * 10:]
height_hist2 = np.concatenate([height_hist2,np.zeros(10-len(height_hist2))])
height_hist2 = height_hist2.reshape(1, 10)
#
height_hist = np.concatenate([height_hist1,height_hist2],axis=0)
height_hist = np.cumsum(height_hist.reshape(-1, 10), axis=1)[:, -1]
plt.bar(height_bins, height_hist[10:], width=10, color='green')

plt.title('Height distribution of male teenagers')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()
