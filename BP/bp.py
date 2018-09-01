#!\usr\bin\python
# coding=UTF-8
import numpy as np
import math
from datetime import *
from scipy.misc import logsumexp
from collections import defaultdict
import random
from matplotlib import pyplot as pt


class Sen:
    def __init__(self,sentence,pos):
        self.sentence=sentence
        self.pos=pos

class NN:
    def __init__(self):
        self.feature = {}  # 特征向量字典
        self.dic_tags = {}  # 词性字典
        self.tags = []  # 词性列表
        self.len_feature = 0  # 特征向量维度
        self.len_tags = 0  # 词性维度
        self.sentences_train = []  # 句子（词语列表）的列表——训练集
        self.pos_train = []  # 各句词性的列表——训练集
        self.dataset_train=[]  #
        self.len_sentences_train = 0  # 句子数量——训练集
        self.sentences_dev = []  # 句子（词语列表）的列表——测试集
        self.pos_dev = []  # 各句词性的列表——测试集
        self.dataset_dev=[] #
        self.len_sentences_dev = 0  # 句子数量——测试集
        self.sentences_test = []
        self.pos_test = []  # 测试集
        self.dataset_test=[] #

        self.output_file = 'big_1.txt'
        self.neurons_2=0
        self.neurons_1=0
        self.neurons_0=0
        self.weight_0=[]
        self.b_1=[]
        self.weight_1=[]
        self.b_2=[]

    def readfile(self, filename):
        sentences = []
        pos = []
        dataset=[]
        with open(filename, 'r') as fr:
            temp_sentence = []
            temp_pos = []
            for line in fr:
                if len(line) > 1:
                    cur_word = line.strip().split()[1].decode('utf-8')
                    cur_tag = line.strip().split()[3]
                    temp_pos.append(cur_tag)
                    temp_sentence.append(cur_word)
                else:
                    sentences.append(temp_sentence)
                    pos.append(temp_pos)
                    sen=Sen(temp_sentence,temp_pos)
                    dataset.append(sen)
                    temp_sentence = []
                    temp_pos = []
        return sentences, pos,dataset

    def read_data(self):
        # self.sentences_train, self.pos_train, self.dataset_train = self.readfile('train.conll')
        # self.sentences_dev, self.pos_dev, self.dataset_dev = self.readfile('dev.conll')
        self.sentences_train, self.pos_train,self.dataset_train = self.readfile('../train.conll')
        self.sentences_dev, self.pos_dev,self.dataset_dev = self.readfile('../dev.conll')
        self.sentences_test, self.pos_test,self.dataset_test = self.readfile('../test.conll')
        self.len_sentences_train = len(self.sentences_train)
        self.len_sentences_dev = len(self.sentences_dev)

    def bi_feature_templates(self, pre_tag):
        return ['01:' + pre_tag]

    def uni_feature_templates(self, sentence, index):
        f = []
        # words_of_sentence
        wos = sentence
        if index == 0:
            wi_minus = '$$'
            ci_minus_minus1 = '$'
        else:
            wi_minus = wos[index - 1]
            ci_minus_minus1 = wi_minus[-1]

        if index == len(wos) - 1:
            wi_plus = '##'
            ci_plus0 = '#'
        else:
            wi_plus = wos[index + 1]
            ci_plus0 = wi_plus[0]
        ci0 = wos[index][0]
        ci_minus1 = wos[index][-1]
        f.append('02:' + wos[index])
        f.append('03:' + wi_minus)
        f.append('04:' + wi_plus)
        f.append('05:' + wos[index] + '*' + ci_minus_minus1)
        f.append('06:' + wos[index] + '*' + ci_plus0)
        f.append('07:' + ci0)
        f.append('08:' + ci_minus1)
        for i in range(1, len(wos[index]) - 1):
            f.append('09:' + wos[index][i])
            f.append('10:' + wos[index][0] + '*' + wos[index][i])
            f.append('11:' + wos[index][-1] + '*' + wos[index][i])
        if len(wos[index]) == 1:
            f.append('12:' + wos[index] + '*' + ci_minus_minus1 + '*' + ci_plus0)
        for i in range(0, len(wos[index]) - 1):
            if wos[index][i] == wos[index][i + 1]:
                f.append('13:' + wos[index][i] + '*' + 'consecutive')
        for i in range(0, 4):
            if i > len(wos[index]) - 1:
                break
            f.append('14:' + wos[index][0:i + 1])
            f.append('15:' + wos[index][-(i + 1)::])
        return f

    def create_feature_templates(self, sentence, index, pre_tag):
        f = self.uni_feature_templates(sentence, index)
        f.extend(self.bi_feature_templates(pre_tag))
        return f

    def create_feature_space(self):
        for tags in self.pos_train:
            for tag in tags:
                if tag not in self.tags:
                    self.tags.append(tag)
        self.dic_tags = {pos: index for index, pos in enumerate(self.tags)}
        self.len_tags = len(self.tags)
        for i in range(self.len_sentences_train):
            sentence = self.sentences_train[i]
            for j in range(len(self.sentences_train[i])):
                f = self.uni_feature_templates(sentence, j)
                for item in f:
                    if item not in self.feature:
                        self.feature[item] = len(self.feature)
        self.len_feature = len(self.feature)
        self.neurons_0=self.len_feature  #?????
        self.neurons_1=16 #int(np.sqrt(self.len_feature))/8  #中间层神经元数量
        self.neurons_2=self.len_tags
        np.random.seed(2)
        self.weight_0 = 2*np.random.random((self.neurons_1,self.neurons_0))-1
        self.b_1=np.random.random((self.neurons_1,1))
        self.weight_1=2*np.random.random((self.neurons_2,self.neurons_1))-1
        self.b_2 = np.random.random((self.neurons_2, 1))

    def sigma(self,z):
        return 1/(1+np.exp(-z))

    def sigma_prime(self,z):
        return self.sigma(z)*(1-self.sigma(z))

    def prime(self,a):
        return a*(1-a)

    def ReLU(self,z):
        return np.where(z>=0,z,0.1*z)

    def ReLU_Prime(self,z):
        return np.where(z>=0,1,0.1)

    def tanh(self,z):
        a=np.exp(z)
        b=np.exp(-z)
        return np.divide((a-b),(a+b))

    def tanh_prime(self,a):
        return 1-a**2

    def train(self,iteration=1000,break_num=50,shuffle=True,batch_size=50,learnrate=0.2,max_entropy=True,step_opt=True,regularization=True):
        max_precesion=0
        max_epoch=0
        count=0
        costs=[]
        its=[]
        print "batch_size="+str(batch_size)+'\n'
        print "Count of Mid Layer=" + str(self.neurons_1) + '\n'
        print "learnrate=" + str(learnrate) + '\n'
        print "Max entropy:"+str(max_entropy)
        global_step = 1
        decay_steps = 5000000
        decay_rate = 0.96
        learn_rate = learnrate
        if regularization:
            lamda=0.00001
        else:
            lamda=0
        pt.ion()
        for it in range(iteration):
            print "Epoch:"+str(it)
            n=0
            b=0
            cost=0
            w1 = np.zeros_like(self.weight_1)
            g = defaultdict(float)
            b1 = np.zeros_like(self.b_1)
            b2 = np.zeros_like(self.b_2)
            if shuffle:
                random.shuffle(self.dataset_train)
            c, t, a = self.test_data(self.dataset_train)
            print 'train正确率：' + str(a)
            c, t, a = self.test_data(self.dataset_dev)
            print 'dev正确率：' + str(a)
            c, t, a = self.test_data(self.dataset_test)
            print 'test正确率：' + str(a)
            if a>max_precesion:
                max_precesion=a
                max_epoch=it
                count=0
            else:
                count+=1
                if count>=break_num:
                    break
            for data in self.dataset_train:
                sen = data.sentence
                tags = data.pos
                for index_word in range(len(sen)):
                    tag=tags[index_word]
                    index_tag=self.dic_tags[tag]
                    feature=self.uni_feature_templates(sen,index_word)
                    index_feature=[self.feature[f] for f in feature if f in self.feature]

                    z1,a1,z2,a2 = self.forward(index_feature)

                    # BP
                    y = np.zeros((self.len_tags, 1))
                    y[index_tag, 0] = 1

                    if max_entropy:
                        cost+=-np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))    # 交叉熵函数
                    else:
                        cost += np.sum((y - a2) * (y - a2))
                    n+=1

                    if max_entropy:
                        dz2 = y - a2
                        dw2 = np.dot(dz2, a1.T)
                        error1 = np.dot(self.weight_1.T, dz2)
                        dz1 = self.prime(a1) * error1
                    else:
                        error2 = y - a2
                        dz2 = self.prime(a2) * error2  # sigmoid function

                        dw2 = np.dot(dz2, a1.T)
                        error1 = np.dot(self.weight_1.T, dz2)
                        dz1 = self.prime(a1) * error1  # sigmoid function
                    b2 += dz2
                    b1 += dz1
                    w1 += dw2
                    for i in index_feature:
                        g[i] += dz1
                    b += 1

                    if b==batch_size:
                        if regularization:
                            self.weight_0*=(1-lamda*learn_rate)
                            self.weight_1 *= (1 - lamda * learn_rate)
                        if step_opt:
                            learn_rate = learnrate * decay_rate ** (global_step / decay_steps)
                        for i, value in g.items():
                            self.weight_0[:, i][:, None] += learn_rate/b * value
                        self.weight_1 += learn_rate/b * w1
                        self.b_1 += learn_rate/b * b1
                        self.b_2 += learn_rate/b * b2
                        b=0
                        w1 = np.zeros_like(self.weight_1)
                        g = defaultdict(float)
                        b1 = np.zeros_like(self.b_1)
                        b2 = np.zeros_like(self.b_2)
                        global_step+=1
            if b>0:
                for i, value in g.items():
                    self.weight_0[:, i][:, None] += learn_rate/b * value
                self.weight_1 += learn_rate/b * w1
                self.b_1 += learn_rate/b * b1
                self.b_2 += learn_rate/b * b2
                w1 = np.zeros_like(self.weight_1)
                g = defaultdict(float)
                b1 = np.zeros_like(self.b_1)
                b2 = np.zeros_like(self.b_2)
                b = 0
                global_step += 1
            cost=cost/n+lamda * (np.sum(self.weight_0 ** 2) + np.sum(self.weight_1 ** 2))
            costs.append(cost)
            its.append(it)

            pt.xlabel("Epoch")
            pt.ylabel("Cost")
            pt.plot(its, costs)
            pt.draw()
            pt.pause(0.01)

        print "Max Iteration:"+str(max_epoch)+"\nmax_precision"+str(max_precesion)

        pt.figure(2)
        pt.xlabel("Epoch")
        pt.ylabel("Cost")
        pt.plot(its, costs)
        pt.show()

    def forward(self,index_feature):
        t=np.sum(self.weight_0[:,index_feature], axis=1)
        z1 = t[:,np.newaxis]+self.b_1
        a1 = self.sigma(z1)  #sigmoid function
        z2 = np.dot(self.weight_1,a1)+self.b_2
        a2 = self.sigma(z2) #sigmoid function
        return z1,a1,z2,a2

    def update(self,index_feature,index_tag):
        # forward
        z1,a1,z2,a2=self.forward(index_feature)
        #n=np.argmax(a2)
        # if n==index_tag:
        #      return
        # BP
        y=np.zeros((self.len_tags,1))
        y[index_tag,0]=1
        error2=y-a2
        dz2=self.prime(a2)*error2
        dw2=np.dot(dz2,a1.T)
        # dw2=np.zeros_like(self.weight_1)
        # dw2[:,n]=-1
        # dw2[:,index_tag]=1
        # dz2=np.dot(a1,dw2)
        error1=np.dot(self.weight_1.T,dz2)
        dz1=self.prime(a1)*error1
        learn_rate=1.0
        self.b_2+=learn_rate*dz2
        self.b_1+=learn_rate*dz1
        self.weight_1+=learn_rate*dw2
        self.weight_0[:,index_feature] += learn_rate*dz1

    def update_batch(self,batch,learnrate):
        w1=np.zeros_like(self.weight_1)
        g=defaultdict(float)
        b1=np.zeros_like(self.b_1)
        b2=np.zeros_like(self.b_2)
        for data in batch:
            sen = data.sentence
            tags = data.pos
            for index_word in range(len(sen)):
                tag = tags[index_word]
                index_tag = self.dic_tags[tag]
                feature = self.uni_feature_templates(sen, index_word)
                index_feature = [self.feature[f] for f in feature if f in self.feature]
                # forward
                z1,a1,z2,a2 = self.forward(index_feature)
                # BP
                y = np.zeros((self.len_tags, 1))
                y[index_tag, 0] = 1
                error2 = y - a2
                dz2 = self.prime(a2) * error2
                dw2 = np.dot(dz2, a1.T)
                error1 = np.dot(self.weight_1.T, dz2)
                dz1 = self.prime(a1) * error1
                b2 += dz2
                b1 += dz1
                w1 += dw2
                for i in index_feature:
                    g[i] += dz1
        for i,value in g.items():
            self.weight_0[:,i][:,None] += learnrate * value
        self.weight_1 += learnrate * w1
        self.b_1 += learnrate * b1
        self.b_2 += learnrate * b2

    def test_data(self,data_set):
        correct=0
        total=0
        for data in data_set:
            sen = data.sentence
            tags = data.pos
            for index_word in range(len(sen)):
                tag = tags[index_word]
                index_tag = self.dic_tags[tag]
                feature = self.uni_feature_templates(sen, index_word)
                index_feature = [self.feature[f] for f in feature if f in self.feature]
                z1,a1,z2,a2 = self.forward(index_feature)
                index_max=np.argmax(a2)
                if index_tag==index_max:
                    correct+=1
                total+=1
        return correct,total,correct*1.0/total
