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