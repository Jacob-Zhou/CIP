#!/usr/bin/ python3
# -*- coding:utf-8 -*-

def CreateTxt(data,txt):
    #创建文本文件
    with open(data,'r',encoding='utf-8') as infile,open(txt,'w',encoding='utf-8') as outfile:
        for line in infile:
            if len(line) > 1 :
                outfile.write("%s" % line.split()[1])
            else:
                outfile.write('\n')

def CreateDict(data,dic):
    #创建字典
    with open(data,'r',encoding='utf-8') as infile,open(dic,'w',encoding='utf-8') as outfile:
        words = set()
        for line in infile:
            if len(line) > 1 :
                words.add(line.split()[1])
        max_len = 0
        for word in words:
            if max_len < len(word):
                max_len = len(word)
            outfile.write("%s\n" % word)
        return max_len , words  

def MM(txt,output,dic):
    with open(txt,'r',encoding='utf-8') as txt,open(output,'w',encoding='utf-8') as out,open(dic,'r',encoding='utf-8') as diction:
        words = set()
        max_len = 0
        for line in diction:
            if len(line.strip()) > max_len:
                max_len = len(line.strip())
            words.add(line.strip())
        for line in txt:
            line = line.strip()
            start = 0
            while start < len(line) :
                end = start + max_len
                while end - start > 1 and line[start:end] not in words:
                    end -= 1                #从后往前一次减小词长
                out.write("%s\n" % line[start:end])
                start = end 
                


def RMM(txt,output,dic):
    with open(txt,'r',encoding='utf-8') as txt,open(output,'w',encoding='utf-8') as out,open(dic,'r',encoding='utf-8') as diction:
        words = set()
        max_len = 0
        for line in diction:
            if len(line.strip()) > max_len:
                max_len = len(line.strip())
            words.add(line.strip())
        for line in txt:
            line = line.strip()
            end = len(line)
            word = list()
            while end > 0:
                start = end - max_len
                if start < 0:
                    start = 0
                while end - start > 1 and line[start:end] not in words:
                    start += 1          #从前往后一次减小词长
                word.insert(0,line[start:end])
                end = start 
            for mword in word:
                out.write("%s\n" % mword)
            

def evaluate(data,output):
    with open(data,'r',encoding='utf-8') as data,open(output,'r',encoding='utf-8') as out:
        All_prec = [line.split()[1] for line in data if len(line) > 1]
        All_reg = [line.strip() for line in out]
        countAll_prec = len(All_prec)
        countAll_reg = len(All_reg)
#        print(countAll_prec,countAll_reg)
        i , j ,countPrec = 0 , 0 , 0
        while i < countAll_prec and j < countAll_reg:
            if All_prec[i] == All_reg[j]:
                countPrec += 1
            else:
                tmp_i , tmp_j = All_prec[i] , All_reg[j]
                while tmp_i != tmp_j:
                    if len(tmp_i) > len(tmp_j):
                        j += 1
                        tmp_j += All_reg[j]
                    elif len(tmp_i) < len(tmp_j):
                        i += 1
                        tmp_i += All_prec[i]
            i += 1
            j += 1
    precision = countPrec / countAll_reg        #正确率
    recall = countPrec / countAll_prec          #召回率
    F = precision * recall * 2 / (precision + recall)       #F值
    print("正确识别词数：%d\n识别出的词总数：%d\n测试集中词总数：%d\n正确率：%f\n召回率：%f\nF值：%f\n" % (countPrec,countAll_reg,countAll_prec,precision,recall,F))
 

if __name__ == '__main__':
    CreateTxt("./data/data.conll","./data/data.txt")
    CreateDict("./data/data.conll","./data/word.dict")
    MM("./data/data.txt","./data/data_MM.out","./data/word.dict")
    RMM("./data/data.txt","./data/data_RMM.out","./data/word.dict")
    print("前向最大匹配分词：")
    evaluate("./data/data.conll","./data/data_MM.out")
    print("后向最大匹配分词：")
    evaluate("./data/data.conll","./data/data_RMM.out")

