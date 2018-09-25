
config = {
    'train_data_file': '../big_data/train.conll',       # 训练集文件
    'dev_data_file': '../big_data/dev.conll',           # 开发集文件
    'test_data_file': '../big_data/test.conll',         # 测试集文件
    'averaged': True,                                   # 是否使用averaged percetron
    'iterator': 100,                                    # 最大迭代次数
    'stop_iterator': 10,                                # 迭代stop_iterator次性能没有提升则结束
    'save_file': '../result/global_linear_model.data',  # 保存模型数据文件
    'thread_num': '2',                                  # 最大线程数
}
