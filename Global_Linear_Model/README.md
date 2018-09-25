# Global Linear Model(全局线性模型)
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./result:
        res_usew_smalldata_v1.txt: 初始版本，小数据测试，使用W作为权重的评价结果
        res_usev_smalldata_v1.txt: 初始版本，小数据测试，使用V作为权重的评价结果
        res_usew_smalldata_v2.txt: 使用部分特征优化后，小数据测试，使用W作为权重的结果
        res_usev_smalldata_v2.txt: 使用部分特征优化后，小数据测试，使用V作为权重的结果
        res_usew_bigdata_v1.txt: 初始版本，大数据测试，使用W作为权重的评价结果
        res_usev_bigdata_v1.txt: 初始版本，大数据测试，使用V作为权重的评价结果
        res_usew_bigdata_v2.txt: 使用部分特征优化后，大数据测试，使用W作为权重的结果
        res_usev_bigdata_v2.txt: 使用部分特征优化后，大数据测试，使用V作为权重的结果
        global_linear_model_w_b_2.data: 特征优化版本，大数据测试,使用W作为权重保存模型
        global_linear_model_v_b_2.data: 特征优化版本，大数据测试,使用V作为权重保存模型
    ./src:
        global_linear-model.py: 初始版本的代码
        global_linear-model_v2.py: 使用特征提取优化后的代码
        config.py: 配置文件，用字典存储每个参数
    ./README.md: 使用说明
    注：保存的模型过大，未上传github

## 二、运行
### 1.运行环境
    python 3
### 2.运行方法
    #配置文件中各个参数
    config = {
        'train_data_file': '../big_data/train.conll',         # 训练集文件
        'dev_data_file': '../big_data/dev.conll',             # 开发集文件
        'test_data_file': '../big_data/test.conll',    	      # 测试集文件
        'averaged': False,                                    # 是否使用averaged percetron
        'iterator': 100,                                      # 最大迭代次数
        'stop_iterator': 10,                                  # 迭代stop_iterator次性能没有提升则结束
        'save_file': '../result/global_linear_model.data',    # 保存模型数据文件
        'thread_num': '2'                                     # 最大线程数
    }
    
    $ cd ./src
    $ python global_linear-model.py                           # 执行初始版本
    $ python global_linear-model_v2.py                        # 执行特征提取优化版本
### 3.参考结果
#### (1)小数据测试

```
训练集：data/train.conll
开发集：data/dev.conll
```

| partial feature | averaged percetron | 迭代次数 | train 准确率 | dev 准确率 | 时间/迭代 |
| :-------------: | :----------------: | :------: | :----------: | :--------: | :-------: |
|        ×        |         ×          |  15/26   |    99.98%    |   86.70%   |    32s    |
|        ×        |         √          |  23/34   |    99.79%    |   87.46%   |    32s    |
|        √        |         ×          |  16/27   |    99.96%    |   87.49%   |    7s     |
|        √        |         √          |  13/24   |    99.73%    |   88.09%   |    7s     |

#### (2)大数据测试

```
训练集：big-data/train.conll
开发集：big-data/dev.conll
测试集：big-data/test.conll
```

| partial feature | averaged percetron | 迭代次数 | train 准确率 | dev 准确率 | test 准确率 | 时间/迭代 |
| :-------------: | :----------------: | :------: | :----------: | :--------: | :---------: | :-------: |
|        ×        |         ×          |  25/36   |    98.95%    |   93.44%   |   93.18%    |   15min   |
|        ×        |         √          |   8/19   |    98.19%    |   94.27%   |   94.10%    |   17min   |
|        √        |         ×          |  30/41   |    99.10%    |   93.66%   |   93.34%    |  3.5min   |
|        √        |         √          |  15/26   |    99.18%    |   94.26%   |   94.12%    |  3.5min   |
