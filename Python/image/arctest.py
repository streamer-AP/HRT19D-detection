"""
=======
Arctest
=======
用来绘制函数 exp(-t)*cos(2pi*t)的图像
其中 plt.plot 用于具体绘制，‘bo’ 为线形。表示蓝色圆圈。
plt.setp 可以对列表或单个对象进行属性设置
markersize 设置圆圈大小
markerfacecolor 设置圆圈的前景色
"""
import matplotlib.pyplot as plt
import numpy as np


def f(t):
    'A damped exponential'
    s1 = np.cos(2 * np.pi * t)
    e1 = np.exp(-t)
    return s1 * e1


t1 = np.arange(0.0, 5.0, .2)

l = plt.plot(t1, f(t1), 'bo')
plt.setp(l, markersize=10)
plt.setp(l, markerfacecolor='#000000')

plt.show()
