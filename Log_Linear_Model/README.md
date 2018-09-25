# Log Linear Model(对数线性模型)
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result:
        res_nopt_smalldata_1.txt: 初始版本，小数据测试
        res_opt_smalldata_1.txt: 初始版本，使用步长优化,小数据测试
        res_nopt_smalldata_2.txt: 特征优化版本，小数据测试
        res_opt_smalldata_2.txt: 特征优化版本，使用步长优化,小数据测试
        res_nopt_bigdata_1.txt: 初始版本，大数据测试
        res_opt_bigdata_1.txt: 初始版本，使用步长优化,大数据测试
        res_nopt_bigdata_2.txt: 特征优化版本，大数据测试
        res_opt_bigdata_2.txt: 特征优化版本，使用步长优化,大数据测试
        log_linear_model_nopt_b_2.data: 特征优化版本，大数据测试保存模型
        log_linear_model_opt_b_2.data: 特征优化版本，使用步长优化,大数据测试保存模型
    ./src:
        log_linear_model.py: 初始版本的代码
        log_linear_model_v2.py: 使用特征提取优化后的代码
        config.py: 配置文件，用字典存储每个参数
    ./README.md: 使用说明
    注：保存的模型过大，未上传github

## 二、运行
### 1.运行环境
    python 3
### 2.运行方法
    #配置文件中各个参数
    config = {
        'train_data_file': '../big_data/train.conll',   #训练集文件
        'dev_data_file': '../big_data/dev.conll',       #开发集文件
        'test_data_file': '../big_data/test.conll',     #测试集文件
        'iterator': 100,                                #最大迭代次数
        'stop_iterator': 10,                            #迭代stop_iterator次性能没有提升则结束
        'batch_size': 50,                               #batch_size
        'regularization': False,                        #是否正则化
        'step_opt': False,                              #是否步长优化（模拟退火）
        'C': 0.0001,                                    #正则化系数
        'eta': 0.5,                                     #初始步长
        'save_file': '../result/log_linear_model.data', #保存模型数据文件
        'thread_num': '2'                               #设置最大线程数
    }
    
    $ cd ./src
    $ python log_linear-model.py                        #执行初始版本
    $ python log_linear-model_v2.py                     #执行特征提取优化版本
### 3.参考结果
#### (1)小数据测试

```
训练集：data/train.conll
开发集：data/dev.conll
```

| partial-feature | 初始步长 | 步长优化 | 迭代次数 | train准确率 | dev准确率 | 时间/迭代 |
| :-------------: | :------: | :------: | :------: | :---------: | :-------: | --------- |
|        ×        |   0.5    |    ×     |  27/38   |   100.00%   |  87.39%   | 32s       |
|        ×        |   0.5    |    √     |  26/37   |   100.00%   |  87.44%   | 29s       |
|        √        |   0.5    |    ×     |  51/62   |   100.00%   |  87.57%   | 4s        |
|        √        |   0.5    |    √     |  10/21   |   100.00%   |  87.56%   | 4s        |

注：由于正则化效果不明显，故未给出正则化实验结果

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

| partial-feature | 初始步长 | 步长优化 | 迭代次数 | train准确率 | dev准确率 | test准确率 | 时间/迭代 |
| :-------------: | :------: | :-------: | :---------: | :--------: | :--------: | :--------: | --------------- |
|        ×        | 0.5  |   ×   | 21/32 |   99.12%   |   93.54%   | 93.28% | 17min |
|        ×        | 0.5 | √ | 20/31 | 99.30% | 93.77% | 93.37% | 17min |
|        √        | 0.5  |   ×   | 23/34 |   99.24%   |   93.55%   |   93.37%   | 2.5min |
|        √        | 0.5 |   √   | 24/35 |   99.35%   | 93.75% |   93.57%   | 2.5min |

