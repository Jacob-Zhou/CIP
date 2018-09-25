# Condition Random Field(条件随机场)
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result:
        res_nopt_smalldata_v1.txt: 初始版本，小数据测试
        res_opt_smalldata_v1.txt: 初始版本，使用模拟退火,小数据测试
        res_nopt_smalldata_v2.txt: 特征优化版本，小数据测试
        res_opt_smalldata_v2.txt: 特征优化版本，使用模拟退火,小数据测试
        res_nopt_bigdata_v1.txt: 初始版本，大数据测试
        res_opt_bigdata_v1.txt: 初始版本，使用模拟退火,大数据测试
        res_nopt_bigdata_v2.txt: 特征优化版本，大数据测试
        res_opt_bigdata_v2.txt: 特征优化版本，使用模拟退火,大数据测试
        crf_nopt_b_2.data: 特征优化版本，不使用模拟退火,大数据保存模型
        crf_opt_b_2.data: 特征优化版本，使用模拟退火,大数据保存模型
    ./src:
        CRF.py: 初始版本的代码
        CRF_v2.py: 使用特征提取优化后的代码
        config.py: 配置文件，用字典存储每个参数
    ./README.md: 使用说明
    注：保存的模型过大，未上传github

## 二、运行
### 1.运行环境
    python 3
### 2.运行方法
    #配置文件中各个参数
    config = {
        'train_data_file': '../big_data/train.conll',    # 训练集文件
        'dev_data_file': '../big_data/dev.conll',        # 开发集文件
        'test_data_file': '../big_data/test.conll',      # 测试集文件
        'iterator': 100,                                 # 最大迭代次数
        'stop_iterator': 10,                             # 迭代stop_iterator次性能没有提升则结束
        'batch_size': 1,                                 # batch_size
        'regularization': False,                         # 是否正则化
        'step_opt': False,                               # 是否步长优化（模拟退火）
        'C': 0.0001,                                     # 正则化系数
        'eta': 0.5,                                      # 初始步长   
        'save_file': '../result/crf.data',               # 保存模型数据文件
        'thread_num': '2'                                # 设置最大线程数
    }
    
    $ cd ./src 
    $ python CRF.py                                      # 执行初始版本
    $ python CRF_v2.py                                   # 执行特征提取优化版本
### 3.参考结果
#### (1)小数据测试

```
训练集：data/train.conll
开发集：data/dev.conll
```

| partial feature | 初始步长 | 步长优化 | 迭代次数 | train准确率 | dev准确率 | 时间/迭代 |
| :-------------: | :------: | :------: | :------: | :---------: | :-------: | :-------: |
|        ×        |   0.5    |    ×     |  58/69   |   100.00%   |  88.68%   |   1min    |
|        ×        |   0.5    |    √     |  23/34   |   100.00%   |  88.66%   |   1min    |
|        √        |   0.5    |    ×     |  21/32   |   100.00%   |  88.95%   |    16s    |
|        √        |   0.5    |    ×     |  12/23   |   100.00%   |  88.93%   |    18s    |

注：由于正则化效果不明显，故未给出正则化实验结果

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

| partial feature | 初始步长 | 步长优化 | 迭代次数 | train准确率 | dev准确率 | test准确率 | 时间/迭代 |
| :-------------: | :------: | :------: | :------: | :---------: | :-------: | :--------: | :-------: |
|        ×        |   0.5    |    ×     |  18/29   |   98.91%    |  93.72%   |   93.57%   |    1h     |
|        ×        |   0.5    |    √     |  20/31   |   99.41%    |  94.10%   |   93.82%   |    1h     |
|        √        |   0.5    |    ×     |  10/21   |   98.92%    |  93.86%   |   93.48%   |   15min   |
|        √        |   0.5    |    √     |  36/47   |   99.55%    |  94.26%   |   93.98%   |   15min   |

