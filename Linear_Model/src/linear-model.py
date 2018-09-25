import numpy as np
import datetime
import random
import pickle
import os
from config import config

def data_handle(filename):
    sentences = []
    sentence = []
    sentence_num = 0
    word_num = 0
    with open(filename,"r",encoding='utf-8') as dataTxt:
        for line in dataTxt:
            if len(line) == 1:
                sentences.append(sentence)
                sentence = []
                sentence_num += 1
            else:
                word = line.split()[1]
                tag = line.split()[3]
                sentence.append((word,tag))
                word_num += 1
    print("文件%s:共%d个句子，%d个词" % (filename,sentence_num,word_num))
    return sentences    #sentences格式：[[(戴相龙,NR),(,),(,)....],[],[]....]


class linear_model(object):
    def __init__(self, train_data_file, dev_data_file):
        self.train_data = data_handle(train_data_file)      #训练数据集
        self.dev_data = data_handle(dev_data_file)          #开发数据集
        tags = set()
        for sentence in self.train_data:
            for word,tag in sentence:
                tags.add(tag)
        self.tags = list(tags)      #tag的列表
        self.N = len(self.tags)     #tag的数目
    
    def create_feature_template(self,sentence,tag,index):    #创建某个tag相对于sentence句子中第index个词的特征
        words = [_word for _word,_tag in sentence]
        words.insert(0,"^")     #插入句首开始符
        words.append("$")       #插入句尾结束符
        feature_template = set()
        feature_template.add("02:" + tag + "*" + words[index+1])
        feature_template.add("03:" + tag + "*" + words[index])
        feature_template.add("04:" + tag + "*" + words[index+2])
        feature_template.add("05:" + tag + "*" + words[index+1] + "*" + words[index][-1])
        feature_template.add("06:" + tag + "*" + words[index+1] + "*" + words[index+2][0])
        feature_template.add("07:" + tag + "*" + words[index+1][0])
        feature_template.add("08:" + tag + "*" + words[index+1][-1])
        for j in range(1,len(words[index+1])-1):
            feature_template.add("09:" + tag + "*" + words[index+1][j])
            feature_template.add("10:" + tag + "*" + words[index+1][0] + "*" + words[index+1][j])
            feature_template.add("11:" + tag + "*" + words[index+1][-1] + "*" + words[index+1][j])
        if len(words[index+1]) == 1:
            feature_template.add("12:" + tag + "*" + words[index+1] + "*" + words[index][-1] + "*" + words[index+2][0])
        for j in range(len(words[index+1])-1):
            if words[index+1][j] == words[index+1][j+1]:
                feature_template.add("13:" + tag + "*" + words[index+1][j] + "*" + "consecutive")
        for j in range(1,min(len(words[index+1]),4)+1):
            feature_template.add("14:" + tag + "*" + words[index+1][0:j])
            feature_template.add("15:" + tag + "*" + words[index+1][-j:])
        return feature_template

    def create_feature_space(self):
        feature_space = set()
        for sentence in self.train_data:
            tags = [tag for word,tag in sentence]
            for i in range(len(sentence)):
                feature_space |= self.create_feature_template(sentence,tags[i],i)
        self.feature_space_list = list(feature_space)
        self.feature_space = {feature:index for index,feature in enumerate(self.feature_space_list)}
        #特征空间是一个字典，格式：{“01:NR*戴相龙":0 , "":1 ,....}
        self.E = len(self.feature_space)    #特征空间的数目

    def get_score(self,features,averaged=False):        #获取某一部分特征的分数
        score = 0
        for feature in features:
            if feature in self.feature_space:
                if averaged:
                    score += self.v[self.feature_space[feature]]
                else:
                    score += self.w[self.feature_space[feature]]
        return score

    def predict(self, sentence, index,averaged=False):      #预测某个sentence第index位置的最高分数的tag
        scores = [self.get_score(self.create_feature_template(sentence, tag, index),averaged) for tag in self.tags]
        return self.tags[np.argmax(scores)]

    def online_training(self, iterator, stop_iterator, save_file, averaged=False):
        self.w = np.zeros(self.E,dtype=int)         
        self.v = np.zeros(self.E,dtype=int)
        self.update_times = np.zeros(self.E, dtype=int)
        update_time = 0
        if averaged:
            print("使用累加特征权重：")
        else:
            print("不使用累加特征权重：")
        max_dev_data_precision = 0
        max_dev_data_precision_index = 0
        for iter in range(iterator):
            print("第%d次迭代：" % (iter+1))
            startime = datetime.datetime.now()
            print("正在打乱训练数据...")
            random.shuffle(self.train_data)
            print("数据已打乱")
            for sentence in self.train_data:
                for i in range(len(sentence)):
                    right_tag = sentence[i][1]
                    max_tag = self.predict(sentence, i, False)
                    if right_tag != max_tag:                #真实值与预测值不一致
                        update_time += 1
                        right_tag_feature = self.create_feature_template(sentence,right_tag,i)
                        for feature in right_tag_feature:
                            feature_index = self.feature_space[feature]
                            last_w_value = self.w[feature_index]
                            self.w[feature_index] += 1
                            self.update_v(feature_index,last_w_value,update_time)
                        max_tag_feature = self.create_feature_template(sentence,max_tag,i)
                        for feature in max_tag_feature:
                            if feature in self.feature_space:
                                feature_index = self.feature_space[feature]
                                last_w_value = self.w[feature_index]
                                self.w[feature_index] -= 1
                                self.update_v(feature_index,last_w_value,update_time)

            #结束本轮迭代，更新全部的v
            for i in range(self.E):
                last_time = self.update_times[i]
                last_w_value = self.w[i]
                self.update_times[i] = update_time
                self.v[i] += (update_time-last_time)*last_w_value

            print("训练集：",end="")
            train_data_precision = self.evaluate(self.train_data,averaged)
            print("开发集：",end="")
            dev_data_precision = self.evaluate(self.dev_data,averaged)
            if dev_data_precision > max_dev_data_precision:
                now_train_data_precision = train_data_precision
                max_dev_data_precision = dev_data_precision
                max_dev_data_precision_index = iter + 1
                self.save(save_file)            #保存模型
            stoptime = datetime.datetime.now()
            time = stoptime - startime 
            print("本轮用时：%s" % str(time))
            if ((iter+1)-max_dev_data_precision_index) > stop_iterator:     #stop_iterator轮性能没有提升
                break
        print("\n共迭代%d轮" % (iter+1))
        print("开发集第%d轮准确率最高:" % max_dev_data_precision_index)
        print("此时训练集准确率:%f" % now_train_data_precision)
        print("此时开发集准确率:%f" % max_dev_data_precision)

    def update_v(self,feature_index,last_w_value,update_time):
        last_time = self.update_times[feature_index]
        self.update_times[feature_index] = update_time
        self.v[feature_index] += (update_time-last_time-1) * last_w_value + self.w[feature_index]


    def evaluate(self,sentences,averaged = False):  #评价数据集正确率
        count_right = 0
        count_all = 0
        for sentence in sentences:
            for i in range(len(sentence)):
                count_all += 1
                right_tag = sentence[i][1]
                max_tag = self.predict(sentence, i, averaged)
                if right_tag == max_tag:
                    count_right += 1
        precision = count_right/count_all
        print("正确词数：%d\t总词数：%d\t正确率%f" % (count_right,count_all,precision))
        return precision

    def save(self,save_file):
        with open(save_file,"wb") as f:
            pickle.dump(self,f)

    @staticmethod
    def load(save_file):
        with open(save_file,"rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    startime = datetime.datetime.now()

    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    averaged = config['averaged']
    iterator = config['iterator']
    stop_iterator = config['stop_iterator']
    save_file = config['save_file']
    thread_num = config['thread_num']

    os.environ['MKL_NUM_THREADS'] = thread_num

    lm = linear_model(train_data_file, dev_data_file)
    lm.create_feature_space()
    lm.online_training(iterator, stop_iterator, save_file, averaged)
    stoptime = datetime.datetime.now()
    time = stoptime - startime
    print("耗时：" +  str(time))
    print('\n加载模型跑测试集：')
    test_data = data_handle(test_data_file)
    test_model = linear_model.load(save_file)
    test_model.evaluate(test_data, averaged)


