# HMM(隐马尔可夫模型)
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result/
    	res_small_data: 小数据测试结果
    	res_big_data: 大数据测试结果
    ./src/
        config.py: 配置文件
        HMM.py: 隐马尔可夫模型代码
    ./HMM-v2.pptx：参考ppt
    ./README.md: 使用说明

## 二、运行
### 1.运行环境
    python 3
### 2.运行方法
    #配置文件中各个参数
    config = {
        'train_data_file': '../big_data/train.conll',  # 训练集数据文件
        'dev_data_file': '../big_data/dev.conll',      # 开发集数据文件
        'test_data_file': '../big_data/dev.conll',     # 测试集数据文件
        'alpha': 0.01                                  # 平滑参数
    }
    $ cd ./src
    $ python HMM.py			                          # 运行程序
### 3.参考结果
#### (1)小数据测试

注：可以修改不同的alpha比较准确率。训练集数据少结果可能不稳定。

```
训练集：../data/train.conll
开发集：../data/dev.conll
```

| alpha | train准确率 | dev准确率 | 执行时间 |
| :---: | :---------: | :-------: | :------: |
|  0.3  |   92.29%    |  75.74%   |    4s    |


#### (2)大数据测试

```
训练集：../big_data/train.conll
开发集：../big_data/dev.conll
测试集：../big_data/test.conll
```

| alpha | train准确率 | dev准确率 | test准确率 | 执行时间 |
| :---: | :---------: | :-------: | :--------: | :------: |
| 0.01  |   93.70%    |  88.35%   |   88.50%   |   4min   |

