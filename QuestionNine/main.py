import numpy as np
from matplotlib.pyplot import plot, show

cash = np.zeros(1000)
cash[0] = 1000
outcome = np.random.binomial(5, 0.5, size=len(cash))

for i in range(1, len(cash)):
    if outcome[i] < 3 and cash[i - 1] != 0:
        cash[i] = cash[i - 1] - 8
    elif outcome[i] < 6 and cash[i - 1] != 0:
        cash[i] = cash[i - 1] + 8
    elif cash[i-1] == 0:
        print(u"输光钱钱了")
        break
    else:
        raise AssertionError("Unexpected outcome " + outcome)
print(outcome.min(), outcome.max())

# 使用Matplotlib绘制cash数组：
plot(np.arange(len(cash)), cash)
show()
