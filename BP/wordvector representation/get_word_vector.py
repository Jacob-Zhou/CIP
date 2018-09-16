import pickle
import numpy as np

dim_vec=100
dic_word={}
index=0
list_vec=[]

with open("../bigdata/embedding/embed.txt","r",encoding="utf-8") as fo:
    for line in fo:
        str=line.strip().split()
        dic_word[str[0]]=index
        index+=1
        vec=[str[i+1] for i in range(0,dim_vec)]
        list_vec.append(np.array(vec))
with open("word.txt","wb") as fo:
    pickle.dump(dic_word,fo)
with open("vector.txt","wb") as fo:
    pickle.dump(list_vec,fo)
