# Linear Model(线性模型)
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result:
        res_usew_v1_smalldata.txt: 初始版本，小数据测试，使用W作为权重的评价结果
        res_usev_v1_smalldata.txt: 初始版本，小数据测试，使用V作为权重的评价结果
        res_usew_v2_smalldata.txt: 使用部分特征优化后，小数据测试，使用W作为权重的结果
        res_usev_v2_smalldata.txt: 使用部分特征优化后，小数据测试，使用V作为权重的结果
        res_usew_v1_bigdata.txt: 初始版本，大数据测试，使用W作为权重的评价结果
        res_usev_v1_bigdata.txt: 初始版本，大数据测试，使用V作为权重的评价结果
        res_usew_v2_bigdata.txt: 使用部分特征优化后，大数据测试，使用W作为权重的结果
        res_usev_v2_bigdata.txt: 使用部分特征优化后，大数据测试，使用V作为权重的结果
        linear_model_b_w_2.data: 使用部分特征优化后，大数据测试，使用W作为权重的保存模型
        linear_model_b_w_2.data: 使用部分特征优化后，大数据测试，使用V作为权重的保存模型
    ./src:
        config.py: 配置文件，用字典存储每个参数
        linear_model.py: 初始版本的代码
        linear_model_v2.py: 使用特征提取优化后的代码
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
        'averaged': True,                               #是否使用averaged percetron
        'iterator': 100,                                #最大迭代次数
        'stop_iterator': 10,                            #迭代stop_iterator次性能没有提升则结束
        'save_file': '../result/linear_model.data',     #保存模型数据文件
        'thread_num': '2'                               #最大线程数
    }
    
    $ cd ./src
    $ python3 linear-model.py                           #执行初始版本
    $ python3 linear-model_v2.py                        #执行特征提取优化版本
### 3.参考结果
#### (1)小数据测试

```
训练集：data/train.conll
开发集：data/dev.conll
```

| partial feature | averaged percetron | 迭代次数 | train 准确率 | dev 准确率 | 时间/迭代 |
| :-------------: | :----------------: | :------: | :----------: | :--------: | :-------: |
|        ×        |         ×          |   8/19   |   100.00%    |   84.55%   |    37s    |
|        ×        |         √          |  14/25   |    98.76%    |   85.84%   |    37s    |
|        √        |         ×          |  12/23   |   100.00%    |   85.92%   |    6s     |
|        √        |         √          |  11/22   |    98.73%    |   85.58%   |    7s     |

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

| partial feature | averaged percetron | 迭代次数 | train 准确率 | dev 准确率 | test 准确率 | 时间/迭代 |
| :-------------: | :----------------: | :------: | :----------: | :--------: | :---------: | :-------: |
|        ×        |         ×          |  17/28   |    98.53%    |   92.88%   |   92.49%    |   14min   |
|        ×        |         √          |  10/21   |    98.20%    |   93.74%   |   93.48%    |   14min   |
|        √        |         ×          |  18/29   |    98.84%    |   93.18%   |   92.71%    |  2.5min   |
|        √        |         √          |   9/20   |    98.58%    |   93.89%   |   93.66%    |  2.5min   |

