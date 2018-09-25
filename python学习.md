# import函数

用于导入其他模块中的对象（函数、类等）

例如想在B.py中使用A.py中的load()函数，有两种方法

1、import A      调用A.load()

2、from A import load      直接调用load()



# 全局变量 

在函数中修改全局变量需要在变量前加global，例如：

```
Money = 2000
def AddMoney():
   global Money
   Money = Money + 1
 
print(Money)
AddMoney()
print(Money)

>>>2000
>>>2001
```



# 列表操作

A = [] 和 A.clear() 的区别：

A = [] 是新开辟一块内存， 假设原来A = [1,2,3]，B = [A] ，那么当A = [] 时B中的值不会改变，而执行A.clear()时不仅A的值改变，B的值也同样发生改变。



# numpy矩阵操作

不同维度向量相加条件：

情况1.两个二维向量相加：A(1 * n) + B(m * 1) = C (m * n)

情况2.多维向量A和B的最后多个维度相等即可相加：A(p * q * r) + B(q * r) = C (p * q * r)。就好比一个低阶的张量加到高阶的张量的每个元素上去。 



# 设置线程

```
import os
os.environ['MKL_NUM_THREADS'] = thread_num
```



# 字典操作

1.字典的获取

```
a = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
a['a']
>>> 1
a.get('a', -1)
>>> 1
a.get('f', -1)		#使用get函数获取，对于字典中不存在的元素，返回-1
>>> -1
```

2.默认字典的使用

```
from collections import defaultdict
import numpy as np
a = defaultdict(float)
a[0, 0] = 6

>>>defaultdict(<class 'float'>, {(0, 0): 6})

b = np.random.randn(2, 3)

>>>[[ 0.46931403 -0.0349126  -1.61533732]
 [ 1.40665706  1.22382829  0.63772504]]

for id, value in a.items():
    b[id] += a[id]

>>>[[ 6.46931403 -0.0349126  -1.61533732]
 [ 1.40665706  1.22382829  0.63772504]]
```



# logsumexp函数


```
from scipy.misc import logsumexp
import numpy as np

a = np.array([10000, 10001, 10002])
b = np.log(np.sum(np.exp(np.array(a))))
>>> inf		#exp时上溢

b = logsumexp(a)
>>> 10002.4076
```



# pickle包


```
import pickle

a = [1,2,3,4,5]
with open('test.txt', 'wb') as f:
    pickle.dump(a, f)		#保存对象（万物皆对象）
with open('test.txt', 'rb') as f:
    b = pickle.load(f)		#加载对象
print(b)
>>> [1,2,3,4,5]
```



# random.shuffle函数


```
import random

random.seed(1)			#设置随机数种子
a = [i for i in range(10)]
>>>[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

random.shuffle(a)		#对一个列表进行打乱
>>>[6, 8, 9, 7, 5, 3, 0, 4, 1, 2]
```