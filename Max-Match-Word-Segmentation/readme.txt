���л�����python 2.7,������ģ��xpinyin��Ϊ�˰Ѵʵ�д���ļ��ﰴƴ��˳�򣬿��п���)

���з�����
  python src/CreateDataTxt.py
  python src/CreateWordDict.py
  python src/WordSplit.py
  python src/Evaluate.py

���н��:
  P=0.99343
  R=0.99066
  F=0.99204

src�ļ���:
  1��CreateWordDict.py
  �õ��˵�������xpinyin,�����ʵ�dict�����׺���Ϊkey�����б�Ϊvalue
  ��ƴ��˳��д��word.dict.txt,
  ��ʽΪ [�׺���1]:[�Ըú���Ϊ�׵Ĵ���1] [����2]......
         [�׺���2]:[�Ըú���Ϊ�׵Ĵ���1] [����2]......

  2��CreateDataTxt.py
  ����ë�ı�,��data.conll�ļ��еĸ�ʽ�޸�Ϊ��ÿ��һ�仰������֮���޿ո�
  ����Ϊdata.txt

  3��WordSplit.py
  ��ȡ�ļ�word.dict.txt
  �����ʵ�dict�����׺���Ϊkey�����б�Ϊvalue
  ���������ƥ���㷨�ִ� ��������out.txt
  ÿ��һ�仰����֮���ÿո����
  
  4��Evaluate.py
  �����㷨����ȷ��

data�ļ���:
  1��data.conll
  ���������ļ�

  2��data.txt
  ë�ı�

  3��word_dict.txt
  �ʵ��ļ�

  4��out.txt
  �ִʵĽ��

