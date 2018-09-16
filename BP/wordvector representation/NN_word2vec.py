#!\usr\bin\python
# coding=UTF-8
import numpy as np
from collections import defaultdict
import random
from matplotlib import pyplot as plt
from scipy.misc import logsumexp
import pickle


class Sen:
    def __init__(self, sentence, pos):
        self.sentence = sentence
        self.pos = pos

class Datasets:
    def __init__(self):
        self.sentences_train = []  # 句子（词语列表）的列表——训练集
        self.pos_train = []  # 各句词性的列表——训练集
        self.dataset_train = []  #
        self.sentences_dev = []  # 句子（词语列表）的列表——测试集
        self.pos_dev = []  # 各句词性的列表——测试集
        self.dataset_dev = []  #
        self.sentences_test = []
        self.pos_test = []  # 测试集
        self.dataset_test = []  #

    def readfile(self, filename):
        sentences = []
        pos = []
        dataset = []
        with open(filename, 'r',encoding='utf-8') as fr:
            temp_sentence = []
            temp_pos = []
            for line in fr:
                if len(line) > 1:
                    cur_word = line.strip().split()[1]
                    cur_tag = line.strip().split()[3]
                    temp_pos.append(cur_tag)
                    temp_sentence.append(cur_word)
                else:
                    sentences.append(temp_sentence)
                    pos.append(temp_pos)
                    sen = Sen(temp_sentence, temp_pos)
                    dataset.append(sen)
                    temp_sentence = []
                    temp_pos = []
        return sentences, pos, dataset

    def get_data(self):
        # self.sentences_train, self.pos_train, self.dataset_train = self.readfile('train.conll')
        # self.sentences_dev, self.pos_dev, self.dataset_dev = self.readfile('dev.conll')
        self.sentences_train, self.pos_train,self.dataset_train = self.readfile('../bigdata/ctb5/train.conll')
        self.sentences_dev, self.pos_dev,self.dataset_dev = self.readfile('../bigdata/ctb5/dev.conll')
        self.sentences_test, self.pos_test,self.dataset_test = self.readfile('../bigdata/ctb5/test.conll')

class Word2Vec:
    def __init__(self,features,len_features,sentence_train):
        self.features=features
        self.len_features=len_features
        self.sentences=sentence_train
        self.dim_vec=50
        self.vectors=2*np.random.random((self.dim_vec,self.len_features))-1
        self.weight_0=np.hstack((self.vectors,self.vectors))
        self.b0=2*np.random.random((self.dim_vec,1))-1
        self.weight_1=2*np.random.random((self.len_features,self.dim_vec))-1
        self.b_1 = 2*np.random.random((self.len_features,1))-1

        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        self.UNKNOWN = 'UNKNOWN'

    def sigma(self, z):
        return 1 / (1 + np.exp(-z))

    def sigma_prime(self, z):
        return self.sigma(z) * (1 - self.sigma(z))

    def prime(self, a):
        return a * (1 - a)

    def softmax(self, z):
        return np.exp(z-logsumexp(z))

    def forward(self,index_feature):
        # index_feature:前一个词的index，后一个词的index(排列后)
        t = np.sum(self.weight_0[:, index_feature], axis=1)
        z1 = t[:, np.newaxis] + self.b0
        a1 = self.sigma(z1)  # sigmoid function
        z2 = np.dot(self.weight_1, a1) + self.b_1
        a2 = self.softmax(z2)  # sigmoid function
        return z1, a1, z2, a2

    def create_feature(self,sen,index_word):
        index_feature=[]
        if index_word == 0:
            index_feature.append(self.features[self.BOS])
        else:
            word=sen[index_word - 1]
            index_feature.append(self.features[word] if word in self.features else self.features[self.UNKNOWN])

        if index_word == len(sen)-1:
            index_feature.append(self.features[self.EOS])
        else:
            word = sen[index_word + 1]
            index_feature.append(self.features[word] if word in self.features else self.features[self.UNKNOWN])
        index_feature[1]+=self.len_features
        return index_feature

    def train(self,epochs=30,shuffle=True,batch_size=1,learnrate=0.5):
        learn_rate=learnrate
        global_step=0
        print("Word2Vec")
        for it in range(epochs):
            print("Epoch:" + str(it))
            right=0
            b = 0
            total = 0
            w1 = np.zeros_like(self.weight_1)
            g = defaultdict(float)
            b1 = np.zeros_like(self.b_1)
            b0 = np.zeros_like(self.b0)
            if shuffle:
                random.shuffle(self.sentences)
            for sen in self.sentences:
                for index_word in range(len(sen)):
                    word=sen[index_word]
                    index_out=self.features[word] if word in self.features else self.features[self.UNKNOWN]
                    index_feature=self.create_feature(sen,index_word)
                    z1, a1, z2, a2 = self.forward(index_feature)

                    index=np.argmax(a2)
                    total+=1
                    if index_out==index:
                        right+=1
                    # BP
                    y = np.zeros((self.len_features, 1))
                    y[index_out, 0] = 1
                    dz2 = y - a2
                    dw2 = np.dot(dz2, a1.T)
                    error1 = np.dot(self.weight_1.T, dz2)
                    dz1 = self.prime(a1) * error1
                    b1 += dz2
                    b0 += dz1
                    w1 += dw2
                    for i in index_feature:
                        g[i%self.len_features] += dz1
                    b += 1

                    if b == batch_size:
                        for i, value in g.items():
                            self.vectors[:, i][:, None] += learn_rate / b * value
                        self.weight_0 = np.hstack((self.vectors, self.vectors))
                        self.weight_1 += learn_rate / b * w1
                        self.b0 += learn_rate / b * b0
                        self.b_1 += learn_rate / b * b1
                        b = 0
                        w1 = np.zeros_like(self.weight_1)
                        g = defaultdict(float)
                        b1 = np.zeros_like(self.b_1)
                        b0 = np.zeros_like(self.b0)
                        global_step += 1
            if b >0:
                for i, value in g.items():
                    self.vectors[:, i][:, None] += learn_rate / b * value
                self.weight_0 = np.hstack((self.vectors, self.vectors))
                self.weight_1 += learn_rate / b * w1
                self.b0 += learn_rate / b * b0
                self.b_1 += learn_rate / b * b1
                b = 0
                w1 = np.zeros_like(self.weight_1)
                g = defaultdict(float)
                b1 = np.zeros_like(self.b_1)
                b0 = np.zeros_like(self.b0)
                global_step += 1
            print("正确率："+str(right/total))

        # 归一化处理
        ceil=np.max(self.vectors)
        floor=np.min(self.vectors)
        self.vectors-=floor
        span=ceil-floor
        self.vectors/=span

        # 文件保存
        with open("word_vector","wb") as f:
            pickle.dump(self.vectors.T,f)

        # with open("word_vector2", "r") as f:
        #     print(pickle.load(f))

class NN:
    def __init__(self):
        self.feature = {}  # 特征向量字典
        self.dic_tags = {}  # 词性字典
        self.tags = []  # 词性列表
        self.len_feature = 0  # 特征向量维度
        self.len_tags = 0  # 词性维度
        self.sentences_train = []  # 句子（词语列表）的列表——训练集
        self.pos_train = []  # 各句词性的列表——训练集
        self.dataset_train = []  #
        self.len_sentences_train = 0  # 句子数量——训练集
        self.sentences_dev = []  # 句子（词语列表）的列表——测试集
        self.pos_dev = []  # 各句词性的列表——测试集
        self.dataset_dev = []  #
        self.len_sentences_dev = 0  # 句子数量——测试集
        self.sentences_test = []
        self.pos_test = []  # 测试集
        self.dataset_test = []  #
        self.indexs_words = [] # 句子所产生的字序列表

        self.BOS='<SOS>'  #'<BOS>'
        self.EOS='<EOS>'  # '<EOS>'
        self.UNKNOWN='UNK'

        # 词向量
        self.vectors=[]
        self.words={}
        self.dim_vector = 100

        self.chains=5
        self.p=0.0
        # 神经网络参数
        self.output_file = 'big_1.txt'
        self.neurons_2 = 0
        self.neurons_1 = 0
        self.neurons_0 = 0
        self.weight_0 = []
        self.b_1 = []
        self.weight_1 = []
        self.b_2 = []

        self.num_layers=2
        self.biases=[]
        self.weights=[]

        # adagrad
        self.epsilon=1e-8
        self.adagrad=False


    def read_data(self):
        datasets=Datasets()
        datasets.get_data()
        self.sentences_train=datasets.sentences_train
        self.pos_train=datasets.pos_train
        self.dataset_train = datasets.dataset_train
        self.sentences_dev=datasets.sentences_dev
        self.pos_dev=datasets.pos_dev
        self.dataset_dev = datasets.dataset_dev
        self.sentences_test=datasets.sentences_test
        self.pos_test =datasets.pos_test
        self.dataset_test = datasets.dataset_test
        self.len_sentences_train = len(self.sentences_train)
        self.len_sentences_dev = len(self.sentences_dev)

    def create_feature_space(self):
        # 读取词向量+字典
        with open("vector.txt", "rb") as f:
            self.vectors=pickle.load(f)
        with open("word.txt", "rb") as f:
            self.words=pickle.load(f)
        self.si=self.words[self.BOS]
        self.ei=self.words[self.EOS]=len(self.words)
        self.vectors.append(2*np.random.random((1,self.dim_vector))-1)
        self.ui=self.words[self.UNKNOWN] = len(self.words)
        self.vectors.append(2 * np.random.random((1, self.dim_vector)) - 1)

        for data in self.dataset_train:
            tags=data.pos
            sen=data.sentence
            for tag in tags:
                if tag not in self.tags:
                    self.tags.append(tag)
            index_words = []
            half = int(self.chains / 2)
            for index in range(-half, half + len(sen)):
                if index < 0:
                    index_words.append(self.si)
                elif index >= len(sen):
                    index_words.append(self.ei)
                else:
                    f = sen[index]
                    # 未知的word，index设为num_words，对此word随机产生其词向量
                    if f in self.words:
                        index_words.append(self.words[f])
                    else:
                        index_words.append(len(self.words))
                        self.words[f] = len(self.words)
                        self.vectors.append(2*np.random.random((1, self.dim_vector))-1)
            self.indexs_words.append(index_words)

        self.indexs_dev=[]
        for sen in self.sentences_dev:
            self.indexs_dev.append(self.create_feature(sen))
        self.indexs_test=[]
        if len(self.sentences_test)>1:
            for sen in self.sentences_test:
                self.indexs_test.append(self.create_feature(sen))

        self.dic_tags = {pos: index for index, pos in enumerate(self.tags)}
        self.len_tags = len(self.tags)

        self.neurons_0 = self.dim_vector * self.chains  # 输入层
        self.neurons_1 = 3 * self.dim_vector
        self.neurons_2 = self.len_tags
        np.random.seed(2)
        self.weight_0 = 2 * np.random.random((self.neurons_1, self.neurons_0)) - 1
        self.b_1 = np.random.random((self.neurons_1, 1))
        self.weight_1 = 2 * np.random.random((self.neurons_2, self.neurons_1)) - 1
        self.b_2 = np.random.random((self.neurons_2, 1))

        if self.adagrad:
            self.gw0 = self.epsilon * np.ones_like(self.weight_0)
            self.gw1 = self.epsilon * np.ones_like(self.weight_1)
            self.gb1 = self.epsilon * np.ones_like(self.b_1)
            self.gb2 = self.epsilon * np.ones_like(self.b_2)
        else:
            self.gw0 = np.ones_like(self.weight_0)
            self.gw1 = np.ones_like(self.weight_1)
            self.gb1 = np.ones_like(self.b_1)
            self.gb2 = np.ones_like(self.b_2)

    def create_feature(self,sen):
        index_feature=[]
        half=int(self.chains/2)
        for index in range(-half,half+len(sen)):
            if index<0:
                index_feature.append(self.si)
            elif index>=len(sen):
                index_feature.append(self.ei)
            else:
                f = sen[index]
                # 未知的word，index设为num_words，对此word随机产生其词向量
                if f in self.words:
                    index_feature.append(self.words[f])
                else:
                    index_feature.append(len(self.words))
                    self.words[f] = len(self.words)
                    self.vectors.append(2 * np.random.random((1, self.dim_vector)) - 1)

        return index_feature

    def input(self,index_feature):
        x=self.vectors[index_feature[0]].reshape(-1,self.dim_vector)
        for i in range(1,len(index_feature)):
            vec=self.vectors[index_feature[i]].reshape(-1,self.dim_vector)
            x=np.concatenate((x,vec),axis=1)
        x = np.array(x, dtype='float64')
        return x

    def sigma(self, z):
        return 1 / (1 + np.exp(-z))

    def sigma_prime(self, z):
        return self.sigma(z) * (1 - self.sigma(z))

    def prime(self, a):
        return a * (1 - a)

    def tanh(self,x):
        ex=np.exp(x)
        ex_=np.exp(-x)
        return (ex-ex_)/(ex+ex_)

    def prime_tanh(self,a):
        return 1-a**2

    def maxout(self, z):
        return np.exp(z-logsumexp(z))

    def maxout_prime(self,a):
        r=np.zeros((len(a),len(a)))
        for i in range(len(a)):
            for j in range(len(a)):
                if i==j:
                    r[i,j]=a[j]*(1-a[j])
                else:
                    r[i,j]=a[i]**2
        return r

    def ReLU(self, z):
        return np.where(z >= 0, z, 0.1 * z)

    def ReLU_Prime(self, z):
        return np.where(z >= 0, 1, 0.1)

    def Adagrad(self,w0,w1,b1,b2):
        self.gw0+=w0**2
        self.gw1+=w1**2
        self.gb1+=b1**2
        self.gb2+=b2**2


    def train(self, iteration=300, break_num=20, shuffle=True, batch_size=1, learnrate=0.2, max_entropy=True,
              step_opt=False, regularization=True,drop_out=False):
        max_precesion = 0
        max_epoch = 0
        count = 0
        costs = []
        precisions_train=[]
        precisions_dev = []

        its = []
        print("batch_size=" + str(batch_size))
        print("Count of Mid Layer=" + str(self.neurons_1))
        print("learnrate=" + str(learnrate))
        print("Max entropy:" + str(max_entropy))
        global_step = 1
        learn_rate = learnrate

        if regularization:
            lamda = 0.1/self.len_sentences_train
        else:
            lamda = 0
        plt.ion()
        for it in range(iteration):
            print("Epoch:" + str(it))
            n = 0
            b = 0
            cost = 0
            w1 = np.zeros_like(self.weight_1)
            w0=np.zeros_like(self.weight_0)
            b1 = np.zeros_like(self.b_1)
            b2 = np.zeros_like(self.b_2)
            g=defaultdict(float)

            allsets=[(indexs,pos) for indexs,pos in zip(self.indexs_words,self.pos_train)]
            if shuffle:
                random.shuffle(allsets)
            # 单轮训练
            for index_words,tags in allsets:
                index_feature=index_words
                xs = self.input(index_feature)
                start=0
                for index in range(len(tags)):
                    tag = tags[index]
                    index_tag = self.dic_tags[tag]
                    x=xs[:,start:start+self.chains*self.dim_vector]
                    start += self.dim_vector
                    z1, a1, z2, a2 = self.forward(x)

                    y = np.zeros((self.len_tags, 1))
                    y[index_tag, 0] = 1
                    # BP
                    if max_entropy:
                        cost += -np.sum(y * np.log(a2+self.epsilon) + (1 - y) * np.log(1 - a2+self.epsilon))  # 交叉熵函数
                    else:
                        cost += np.sum((y - a2) * (y - a2))
                    n += 1

                    if max_entropy:
                        dz2 =y- a2
                        dw2 = np.dot(dz2, a1.T)
                        error1 = np.dot(self.weight_1.T, dz2)
                        dz1 = self.prime(a1) * error1
                        dw1=np.dot(dz1,x)
                        # error0 = np.dot(self.weight_0.T, dz1)
                    else:
                        error2 = y- a2
                        dz2 = self.prime(a2) * error2  # sigmoid function
                        dw2 = np.dot(dz2, a1.T)
                        error1 = np.dot(self.weight_1.T, dz2)
                        dz1 = self.prime(a1) * error1  # sigmoid function
                        dw1 = np.dot(dz1, x)
                    b2 += dz2
                    b1 += dz1
                    w1 += dw2
                    w0 += dw1
                    # position=0
                    # for i in index_feature[index:index+self.chains]:
                    #     g[i]+=error0[position:position+self.dim_vector]
                    #     position+=self.dim_vector
                    b += 1

                    if b == batch_size:
                        if regularization:
                            self.weight_0 *= (1 - lamda * learn_rate)
                            self.weight_1 *= (1 - lamda * learn_rate)
                        if self.adagrad:
                            self.Adagrad(w0,w1,b1,b2)
                        if self.adagrad:
                            self.weight_0 += learn_rate / b * w0 / np.sqrt(self.gw0)
                            self.weight_1 += learn_rate / b * w1 / np.sqrt(self.gw1)
                            self.b_1 += learn_rate / b * b1 / np.sqrt(self.gb1)
                            self.b_2 += learn_rate / b * b2 / np.sqrt(self.gb2)
                        else:
                            self.weight_0 += learn_rate / b * w0
                            self.weight_1 += learn_rate / b * w1
                            self.b_1 += learn_rate / b * b1
                            self.b_2 += learn_rate / b * b2
                        # for i,vec in g.items():
                        #     self.vectors[i]=np.array(self.vectors[i],'float64')+vec.T
                        # g=defaultdict(float)
                        b = 0
                        w0 = np.zeros_like(self.weight_0)
                        w1 = np.zeros_like(self.weight_1)
                        b1 = np.zeros_like(self.b_1)
                        b2 = np.zeros_like(self.b_2)
                        global_step += 1
            if b > 0:
                if self.adagrad:
                    self.weight_0 += learn_rate / b * w0 / np.sqrt(self.gw0)
                    self.weight_1 += learn_rate / b * w1 / np.sqrt(self.gw1)
                    self.b_1 += learn_rate / b * b1 / np.sqrt(self.gb1)
                    self.b_2 += learn_rate / b * b2 / np.sqrt(self.gb2)
                else:
                    self.weight_0 += learn_rate / b * w0
                    self.weight_1 += learn_rate / b * w1
                    self.b_1 += learn_rate / b * b1
                    self.b_2 += learn_rate / b * b2
                for i, vec in g.items():
                    self.vectors[i] = np.array(self.vectors[i], 'float64') + vec.T
                g=defaultdict(float)
                w0 = np.zeros_like(self.weight_0)
                w1 = np.zeros_like(self.weight_1)
                b1 = np.zeros_like(self.b_1)
                b2 = np.zeros_like(self.b_2)
                b = 0

                # 评估
            c, t, a = self.test_data(self.indexs_words, self.pos_train)
            precisions_train.append(a)
            print('train正确率：' + str(a))
            c, t, a = self.test_data(self.indexs_dev, self.pos_dev)
            precisions_dev.append(a)
            print('dev正确率：' + str(a))
            if len(self.dataset_test) > 0:
                c, t, a = self.test_data(self.indexs_test, self.pos_test)
            print('test正确率：' + str(a))

                # 迭代停止条件
            if a > max_precesion:
                max_precesion = a
                max_epoch = it
                count = 0
            else:
                count += 1
                if count >= break_num:
                    break
            # 计算损失
            cost = cost / n #+ lamda * (np.sum(self.weight_0 ** 2) + np.sum(self.weight_1 ** 2))
            costs.append(cost)
            its.append(it)
            # 作图规范
            plt.figure(1)
            plt.subplot(211)
            plt.ylabel("Cost")
            plt.plot(its, costs)
            plt.subplot(212)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.plot(its, precisions_train, '-.', its, precisions_dev, 'o-')
            plt.legend(["train", "dev"])
            plt.draw()
            plt.pause(0.01)

        print("Max Iteration:" + str(max_epoch) + "\nmax_precision" + str(max_precesion))
        plt.ioff()
        plt.show()

    def forward(self, x):
        p = self.p
        weight_1=self.weight_1
        t = np.dot(self.weight_0,x.T)
        z1 = t + self.b_1
        a1 = self.sigma(z1)  # sigmoid function
        z2 = np.dot(weight_1, a1) + self.b_2
        a2 = self.sigma(z2)  # sigmoid function
        return z1, a1, z2, a2

    def test_data(self, indexs_,tags_):
        correct = 0
        total = 0
        for indexs,tags in zip(indexs_,tags_):
            xs = self.input(indexs)
            for index_word in range(len(tags)):
                tag = tags[index_word]
                index_tag = self.dic_tags[tag]
                start = index_word * self.dim_vector
                x=xs[:,start:start+self.chains*self.dim_vector]
                z1, a1, z2, a2 = self.forward(x)
                index_max = np.argmax(a2)
                if index_tag == index_max:
                    correct += 1
                total += 1
        return correct, total, correct * 1.0 / total


