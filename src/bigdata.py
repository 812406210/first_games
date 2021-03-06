# coding: utf-8
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import jieba
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold


###########################词袋模型特征############################################
#重组为新的句子
def clean_text(text):
    """
    去掉html标签、移除标点、切分成词/token、去掉停用词、重组为新的句子
    :param text:
    :return:
    """
    print(text)
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    stopwords = {}.fromkeys([line.rstrip() for line in open('../stopwords/stopwords_english.txt')])
    eng_stopwords = set(stopwords)
    print(eng_stopwords)
    words = [w for w in words if w not in eng_stopwords]
    print(words)
    return ' '.join(words)


#   重组为新的句子
def clean_chinese_text(text):
    """
    去掉html标签、移除标点、切分成词/token、去掉停用词、重组为新的句子
    :param text:
    :return:
    """
    #   print(text)
    words = " ".join(jieba.cut(text))
    stopwords = {}.fromkeys([line.rstrip() for line in open('../data/stopwords_chineses.txt',encoding='utf-8')])
    chi_stopwords = set(stopwords)
    #   print(chi_stopwords)
    words = [w for w in words if w not in chi_stopwords]
    #   print(words)
    return ' '.join(words)


def save_train_to_csv(y_test,y_pred):
    save_df = pd.DataFrame(y_test)
    save_df['y_pred_label'] = pd.Series(y_pred).values
    save_df.to_csv('../data/train_result.csv')


def save_test_to_csv(y_pred,ids):
    save_df = pd.DataFrame(y_pred)
    save_df['id'] = pd.Series(ids).values
    save_df.to_csv('../data/test_result.csv')


def save_train_model(LR_model):
    with open('../data/naive_bayes.pickle', 'wb') as fw:
        pickle.dump(LR_model, fw)


def read_csv_file():
    dfSet = pd.read_csv('../data/Train_DataSet.csv', sep=',', escapechar='\\')  # id 7345 , title 7342, content 7266
    dfLabel = pd.read_csv('../data/Train_DataSet_Label.csv', sep=',', escapechar='\\')  # 7355
    dfSet = pd.merge(dfSet, dfLabel, on='id')  # 7340
    return dfSet


def train_model():
    # 1、读取数据,数据合并--->通过pd.merge函数合并，合并的条件id相等
    dfSet = read_csv_file()

    # 2、数据清洗--->a)清除nan数据 ; b)分词 ; c)归一化处理
    df = dfSet.dropna(axis=0, how='all')
    # 数据清洗,对df中的每一个Serial进行清洗
    df['content'] = df['content'].astype(str).apply(clean_chinese_text)
    # 抽取bag of words特征(用sklearn的CountVectorizer)
    vectorizer = CountVectorizer(max_features=5000, token_pattern=u"(?u)\\b\\w+\\b")
    train_data_features = vectorizer.fit_transform(df['content']).toarray()
    print(train_data_features, type(train_data_features))

    # 特征选择
    choose_feature = VarianceThreshold(threshold=0.0)
    train_data_features = choose_feature.fit_transform(train_data_features)
    # 归一化处理
    train_data_features = normalize_handle(train_data_features)


    #  3、模型保存与预测  a)数据切分; b)模型训练; c) 保存与测试结果
    X_train, X_test, y_train, y_test = train_test_split(train_data_features, df.label, test_size=0.2, random_state=0)
    print(X_train, X_test, y_train, y_test)
    # ### 训练分类器
    clf = MultinomialNB(alpha=1.0)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # 保存模型
    save_train_model(clf)
    save_train_to_csv(y_test, y_pred)

    print("准确率为：",((y_pred==y_test).sum())/y_test.shape[0])
    print("准确率为：", accuracy_score(y_test,y_pred))


def normalize_handle(data):
    """
    归一化处理
    错误1：setting an array element with a sequence. 表示列没有对齐
    :return:
    """
    m = MinMaxScaler(feature_range=(0, 4))
    data = m.fit_transform(data)
    return data


def test_predict():
    test_df = pd.read_csv('../data/Test_DataSet.csv', sep=',', escapechar='\\') #7355
    df = test_df.dropna(axis=0, how='all')#.head(100) #7355
    df['content'] = df['content'].astype(str).apply(clean_chinese_text)
    vectorizer = CountVectorizer(max_features=5000, token_pattern=u"(?u)\\b\\w+\\b")
    train_data_features = vectorizer.fit_transform(df['content']).toarray()
    # 特征选择
    choose_feature = VarianceThreshold(threshold=0.0)
    train_data_features = choose_feature.fit_transform(train_data_features)
    # 归一化处理
    train_data_features = normalize_handle(train_data_features)
    print(train_data_features)
    with open('../data/naive_bayes.pickle', 'rb') as fr:
        new_lr = pickle.load(fr)
        test_result = new_lr.predict(train_data_features)
        print("*"*20)
        print(type(test_result))
        save_test_to_csv(test_result,df['id'])


if __name__=='__main__':
    #train_model()
    test_df = pd.read_csv('../data/Test_DataSet.csv', sep=',', escapechar='\\').head(50)
    dfSet = pd.read_csv('../data/Train_DataSet.csv', sep=',', escapechar='\\').head(50)
    li = [dfSet, test_df]
    dfSet = pd.concat(li)

    print(dfSet.describe())



    #   test_predict()


