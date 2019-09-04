# coding: utf-8
import re
from _ast import Lambda

import keras
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras import Model, Input
from keras.backend import concatenate
from keras.models import load_model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Concatenate
from numpy.ma import squeeze
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import jieba
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedShuffleSplit
import  tensorflow as tf
###########################词袋模型特征############################################
#重组为新的句子

def clean_chinese_text(text):
    """
    移除标点、切分成词/token、去掉停用词、重组为新的句子
    :param text:
    :return:
    """
    #print(text)
    words = " ".join(jieba.cut(text))
    stopwords = {}.fromkeys([line.rstrip() for line in open('../data/stopwords_chineses.txt',encoding='utf-8')])
    chi_stopwords = set(stopwords)
    #print(chi_stopwords)
    words = [w for w in words if w not in chi_stopwords]
    #print(words)
    return ' '.join(words)

def read_csv_file():
    dfSet = pd.read_csv('../data/Train_DataSet.csv', sep=',', escapechar='\\')#.head(50)# id 7345 , title 7342, content 7266
    dfLabel = pd.read_csv('../data/Train_DataSet_Label.csv', sep=',', escapechar='\\')#.head(50)# 7355
    dfSet = pd.merge(dfSet, dfLabel, on='id')  # 7340
    return dfSet

def word2vec():
    # 1、读取数据,数据合并--->通过pd.merge函数合并，合并的条件id相等
    dfSet =read_csv_file()

    test_df = pd.read_csv('../data/Test_DataSet.csv', sep=',', escapechar='\\')#.head(50) # 7355
    # 2、数据清洗--->a)清除nan数据 ; b)分词 ; c)归一化处理
    df = dfSet.dropna(axis=0, how='all')
    # 数据清洗,对df中的每一个Serial进行清洗
    df['content'] = df['content'].astype(str).apply(clean_chinese_text)
    test_df['content'] = test_df['content'].astype(str).apply(clean_chinese_text)
    # 抽取bag of words特征(用sklearn的CountVectorizer)
    vectorizer = CountVectorizer(max_features = 6000, token_pattern=u"(?u)\\b\\w+\\b")
    train_data_features = vectorizer.fit_transform(pd.concat([df['content'], test_df['content']],axis=0,ignore_index=True)).toarray()
    print(train_data_features)
    tfidftransformer = TfidfTransformer()
    train_data_features = tfidftransformer.fit_transform(vectorizer.fit_transform(pd.concat([df['content'], test_df['content']],axis=0,ignore_index=True)))
    return train_data_features


def load_model_predict():
    model = load_model("../model/textcnn.h5")
    return model


def TextCNN_model_1(x_train_padded_seqs, y_train, x_test_padded_seqs, y_test):

    #####################训练数据和测试数据进行模型训练####################
    main_input = Input(shape=(50,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn = Concatenate()([cnn1, cnn2, cnn3])
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(3, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=500, epochs=20)
    model.save("../model/textcnn.h5")  #保存keara模型
    # y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    print("result_labels:",result_labels.size,type(result_labels),result_labels)
    y_predict = list(map(int, result_labels))
    print("y_predict",len(y_predict),type(y_predict), y_predict)
    print("y_test",y_test.size,type(y_test),y_test)
    print("准确率为：", ((y_predict == y_test).sum()) / y_test.shape[0])
    print('准确率', accuracy_score(y_test.values, y_predict))


def train_test():
    # 1、读取数据,数据合并--->通过pd.merge函数合并，合并的条件id相等
    dfSet =read_csv_file()
    test_df = pd.read_csv('../data/Test_DataSet.csv', sep=',', escapechar='\\')#.head(50) # 7355
    # 2、数据清洗--->a)清除nan数据 ; b)分词 ; c)归一化处理
    df = dfSet.dropna(axis=0, how='all')
    # 数据清洗,对df中的每一个Serial进行清洗
    df['content'] = df['content'].astype(str).apply(clean_chinese_text)
    test_df['content'] = test_df['content'].astype(str).apply(clean_chinese_text)

    df['content']  = pd.concat([df['content'], test_df['content']], axis=0, ignore_index=True)
    print(df.describe())
    return df

def save_test_to_csv(ids,y_pred):
    save_df = pd.DataFrame(y_pred)
    save_df['id'] = pd.Series(ids).values
    save_df.to_csv('../data/testcnn_result.csv')

if __name__ == '__main__':
    #获取训练集
    # 1、读取数据,数据合并--->通过pd.merge函数合并，合并的条件id相等
    dfSet = pd.read_csv('../data/Train_DataSet.csv', sep=',', escapechar='\\')#.head(50)  # id 7345 , title 7342, content 7266
    dfLabel = pd.read_csv('../data/Train_DataSet_Label.csv', sep=',', escapechar='\\')#.head(50)  # 7355
    dfSet = pd.merge(dfSet, dfLabel, on='id')  # 7340
    df = dfSet.dropna(axis=0, how='all')
    #读取测试数据
    test_df = pd.read_csv('../data/Test_DataSet.csv', sep=',', escapechar='\\')#.head(50)  # 7355

    # 2、数据清洗，将测试数据content和训练数据content合并
    # 数据清洗,对df中的每一个Serial进行清洗
    df['content'] = df['content'].astype(str).apply(clean_chinese_text)
    test_df['content'] = test_df['content'].astype(str).apply(clean_chinese_text)
    all_content = pd.concat([df['content'], test_df['content']], axis=0, ignore_index=True)

    tokenizer = Tokenizer()  # 创建一个Tokenizer对象
    # fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    tokenizer.fit_on_texts(all_content)
    vocab = tokenizer.word_index  # 得到每个词的编号 [0:7340]
    x_train, x_test, y_train, y_test = train_test_split(all_content[0:7340], df['label'], test_size=0.1)
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    # 序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=50)  # 将超过固定值的部分截掉，不足的在最前面用0填充
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)

    print(x_train_padded_seqs.size,y_train.size,x_test_padded_seqs.size,y_test.size) #330300 6606 36700 734
    TextCNN_model_1(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test)

    ######################测试数据，模型测试###################################
    print("df['content']",len(df['content']))
    test_data = all_content[7340:] #[7340:]
    print("test_data:",test_data.size,type(test_data))
    test_word_ids  = tokenizer.texts_to_sequences(test_data)
    test_word_squences = pad_sequences(test_word_ids, maxlen=50)
    model = load_model_predict()
    test_result = model.predict(test_word_squences) #测试数据预测
    print("test_result",type(test_result),len(test_result))
    test_result_labels = np.argmax(test_result, axis=1)  # 获得最大概率对应的标签
    print("result_labels:",test_result_labels.size,type(test_result_labels),test_result_labels)
    y_predict = list(map(int, test_result_labels))
    print("y_predict:",len(y_predict),y_predict)
    print("test_df['id']:",test_df['id'].size,test_df['id'])
    save_test_to_csv(test_df['id'],y_predict) #保存提交数据



