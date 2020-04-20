#coding:utf-8
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# 1 读取数据
train_id = []
train_text = []
with open('../input/Train_DataSet.csv',encoding='utf8') as f:
    for idx,i in enumerate(f):
        if idx == 0:
            pass
        else:
            ik = str(i).split(',')[0]
            i = str(i).split(',')[1:]
            #cop = re.compile("[^\u4e00-\u9fa5^.^，^\d]")#^a-z^A-Z保留中文，逗号，句号，和数字
            #ret = cop.sub('',','.join(i)) # '[A-Za-z]+',
            #ret = re.sub('\d{8,}','',ret) 
            rat = re.sub('[^\u4e00-\u9fa5]{8,}','',','.join(i)) # 非汉字连续超过8个删除
            train_id.append(ik)
            train_text.append(rat.replace('\n','').replace('\\r\\n','').replace(' ','').replace('.',''))
train = pd.DataFrame()
train['id'] = train_id
train['text'] = train_text
train_label = pd.read_csv('../input/Train_DataSet_Label.csv',sep=',')
train = pd.merge(train,train_label,on=['id'],copy=False)

test_id = []
test_text = []
with open('../input/Test_DataSet.csv',encoding='utf8') as f:
    for idx,i in enumerate(f):
        if idx == 0:
            pass
        else:
            ik = str(i).split(',')[0]
            i = str(i).split(',')[1:]
            rat = re.sub('[^\u4e00-\u9fa5]{8,}','',','.join(i)) # 非汉字连续超过8个删除
            test_id.append(ik)
            test_text.append(rat.replace('\n','').replace('\\r\\n','').replace(' ','').replace('.',''))
test = pd.DataFrame()
test['id'] = test_id
test['text'] = test_text
## 2 准备数据
data = []
for text,label in zip(train['text'],train['label']) :
    data.append((text,label))
# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
# 文本长度
def seq_padding(X,padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x,[padding]*(ML-len(x))]) if len(x)<ML else x for x in X
    ])

# 定义数据生成器
class  data_generator:
    def __init__(self,data,batch_size=24):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size !=0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1,X2,Y=[],[],[]
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1,x2 = tokenizer.encode(first=text)
                y=d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1,X2],Y
                    [X1, X2, Y] = [], [], []

## 3 导入bert模型
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import os
import codecs

maxlen = 100
config_path = '../input/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../input/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../input/chinese_L-12_H-768_A-12/vocab.txt'
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
## 继承bert的Tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
tokenizer = OurTokenizer(token_dict)
## 加载模型
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(3, activation='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
    metrics=['sparse_categorical_accuracy']
)
## 4 训练分类
train_D = data_generator(train_data)
valid_D = data_generator(valid_data)
model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
# 测试集
def get_test_data():
    test_data = []
    for i in range(len(test)):
        test_data.append(test['text'][i])
    idxs = list(range(len(test_data)))
    X1 = []
    X2 = []
    for i in idxs:
        d = test_data[i]
        text = d[:maxlen]
        x1,x2 = tokenizer.encode(first=text)
        X1.append(x1)
        X2.append(x2)
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)                    
    test_data_all = [X1,X2]
    return test_data_all
test_data_all = get_test_data()
predictions = model.predict(test_data_all)
result = pd.DataFrame()
result['id'] = test['id']
result['label'] = np.argmax(predictions,axis=1)
result.to_csv('../output/baseline1_20190826.csv',index=False)