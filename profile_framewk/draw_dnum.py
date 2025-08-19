import matplotlib.pyplot as plt
import numpy as np

# 数据
dnum = [1, 2, 3, 4, 6, 8, 12, 24]
NTT = [36.6, 42.6, 48.2, 58.6, 62.6, 69.2, 72.1, 73]
BaseConv = [55, 48, 42, 31.2, 26, 18, 14, 11.835]
Modmul = [7.3, 8.3, 8.7, 9.1, 10.3, 11.7, 12.8, 14.065]
Others = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

# 堆叠柱状图
bar_width = 0.6
x = np.arange(len(dnum))

plt.bar(x, NTT, width=bar_width, label='NTT')
plt.bar(x, BaseConv, width=bar_width, bottom=NTT, label='BaseConv')
plt.bar(x, Modmul, width=bar_width, bottom=np.array(NTT)+np.array(BaseConv), label='Modmul')
plt.bar(x, Others, width=bar_width, bottom=np.array(NTT)+np.array(BaseConv)+np.array(Modmul), label='Others')

# 设置坐标轴
plt.xticks(x, dnum)
plt.xlabel('dnum')
plt.ylabel('Value')
plt.title('Breakdown by dnum')
plt.legend()

plt.tight_layout()
plt.show()
