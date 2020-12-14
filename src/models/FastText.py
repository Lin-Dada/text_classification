# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, g_config):
        self.model_name = g_config.model_name
        self.emb_path = g_config.emb_path
        self.train_path = g_config.train_path                               # 训练集
        self.dev_path = g_config.dev_path                                   # 验证集
        self.test_path = g_config.test_path                                 # 测试集
        self.class_list = g_config.class_list                               # 类别名单
        self.vocab_path = './temp/vocab_fasttext.dict'                                # 词表
        self.save_path = g_config.save_path                                 # 模型训练结果
        self.log_path = g_config.log_path
        self.embedding_pretrained = None                                       # 不需要预训练词向量
        self.device = g_config.device   # 设备

        self.dropout = g_config.dropout                                             # 随机失活
        self.require_improvement = g_config.require_improvement                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = g_config.num_classes                        # 类别数
        self.n_vocab = g_config.n_vocab                                               # 词表大小，在运行时赋值
        self.num_epochs = g_config.num_epochs                                            # epoch数
        self.batch_size = g_config.batch_size                                          # mini-batch大小
        self.pad_size = g_config.pad_size                                            # 每句话处理成的长度(短填长切)
        self.learning_rate = g_config.learning_rate                                     # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 256                                          # 隐藏层大小
        self.n_gram_vocab = 250499                                      # ngram 词表大小


'''Bag of Tricks for Efficient Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x: (x, seq_len, bigram, trigram), y
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
