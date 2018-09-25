import numpy as np
import datetime
from config import config




def data_handle(data):  
    sentence = list()
    sentences = list()  
    with open(data,'r',encoding='utf-8') as dataTxt:
        for line in dataTxt:
            if len(line) > 1:
                word_and_tag = (line.split()[1] , line.split()[3])
                sentence.append(word_and_tag)
            else:
                sentences.append(sentence)
                sentence = []     # 注意sentence = [] 和 sentence.clear()的区别 ,
                                  # sentence = []是新开辟内存，原来的sentence在sentences中的值不会改变，
                                  # 而sentence.clear()直接把原来的sentence清空，使得snetences中的内容也清空了
    return sentences     


class HMM(object):
    def __init__(self,sentences):
        words = set()
        tags = set()
        self.sentences = sentences
        for sentence in sentences:
            for word,tag in sentence:
                words.add(word)
                tags.add(tag)
        words_list = list(words)
        tags_list = list(tags)
        tags_list.append("*start")   #加入开始词性
        tags_list.append("*stop")    #加入结束词性
        words_list.append("???")     #加入未知词
        self.words = {word:index for index,word in enumerate(words_list)} 
        self.tags = {tag:index for index,tag in enumerate(tags_list)}
        self.M = len(self.words)     #词的个数,包括未知词
        self.N = len(self.tags)      #词性个数,包括开始词性和结束词性
        self.transport_matrix = np.zeros((self.N-1,self.N-1))       #最后一行表示从开始词性转移到各词性,最后一列表示转移到结束词性
        self.launch_matrix = np.zeros((self.N-2,self.M))            #最后一列表示发射到未知词

    def launch(self,alpha):         #计算发射矩阵
        for sentence in self.sentences:
            for word,tag in sentence:
                self.launch_matrix[self.tags.get(tag)][self.words.get(word)] += 1
        for i in range(len(self.launch_matrix)):
            sum_line = sum(self.launch_matrix[i])
            for j in range(len(self.launch_matrix[i])):
                self.launch_matrix[i][j] = (self.launch_matrix[i][j] + alpha) / (sum_line + alpha * self.M)

    def transport(self,alpha):      #计算转移矩阵
        for sentence in self.sentences:
            pre = -1
            for word,tag in sentence:
                self.transport_matrix[pre][self.tags.get(tag)] += 1
                pre = self.tags.get(tag)
            self.transport_matrix[pre][-1] += 1
        for i in range(len(self.transport_matrix)):
            sum_line = sum(self.transport_matrix[i])
            for j in range(len(self.transport_matrix[i])):
                self.transport_matrix[i][j] = (self.transport_matrix[i][j] + alpha) / (sum_line + alpha * (self.N-1))

    def viterbi(self, sentence):
        word_index = [self.words.get(word,self.words["???"]) for word in sentence]
        observeNum = len(sentence)                  #句子长度
        tagNum = self.N - 2                         #词性数
        max_p = np.zeros((observeNum, tagNum))      #第一行用于初始化,max_p[i][j]表示从开始到第i个观测对应第j个词性的概率最大值
        path = np.zeros((observeNum, tagNum),dtype="int")       #第一行用于初始化,path[i][j]表示从开始到第i个观测对应第j个词性概率最大时i-1个观测的词性索引值

        transport_matrix = np.log(self.transport_matrix)    #对数处理后，点乘运算变为加法运算
        launch_matrix = np.log(self.launch_matrix)

        path[0] = -1
        max_p[0] = transport_matrix[-1,:-1] + launch_matrix[:,word_index[0]]

        for i in range(1, observeNum):
            probs = transport_matrix[:-1,:-1] + max_p[i-1].reshape(-1,1) + launch_matrix[:,word_index[i]]       #!这一步是关键
            max_p[i] = np.max(probs, axis=0)
            path[i] = np.argmax(probs, axis=0)

        max_p[-1] += transport_matrix[:-1,-1]

        step = np.argmax(max_p[-1])
        gold_path = [step]

        for i in range(observeNum-1, 0, -1):
            step = path[i][step]
            gold_path.insert(0, step)
        return gold_path
    

    def evaluate(self, data):
        total_words = 0
        correct_words = 0
        sentence_num = 0
        print('正在评估数据集...')
        for sentence in data:
            sentence_num += 1
            word_list = []
            tag_list = []
            for word, tag in sentence:
                word_list.append(word)
                tag_list.append(tag)
            predict = self.viterbi(word_list)
            total_words += len(sentence)
            for i in range(len(predict)):
                if predict[i] == self.tags.get(tag_list[i]):
                    correct_words += 1
        print('共%d个句子' % (sentence_num))
        print('共%d个单词，预测正确%d个单词' % (total_words, correct_words))
        print('准确率：%f' % (correct_words / total_words))

if __name__ == "__main__":
    startTime = datetime.datetime.now()

    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    alpha = config['alpha']

    print("train HMM...")
    print("alpha=%f" % config['alpha'])
    sentences = data_handle(train_data_file)
    hmm = HMM(sentences)
    hmm.launch(alpha)
    hmm.transport(alpha)
    # print("\n训练集：")
    # train_data = data_handle(train_data_file)
    # hmm.evaluate(train_data)
    print("\n开发集：")
    dev_data = data_handle(dev_data_file)
    hmm.evaluate(dev_data)
    print("\n测试集")
    test_data = data_handle(test_data_file)
    hmm.evaluate(test_data)
    stopTime = datetime.datetime.now()
    print("\n用时：" + str(stopTime-startTime))




