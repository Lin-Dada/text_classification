# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

class Config(object):

    """配置参数"""
    def __init__(self, g_config):
        self.model_name = g_config.model_name
        self.emb_path = g_config.emb_path
        self.train_path = g_config.train_path                               # 训练集
        self.dev_path = g_config.dev_path                                   # 验证集
        self.test_path = g_config.test_path                                 # 测试集
        self.class_list = g_config.class_list                               # 类别名单
        self.vocab_path = g_config.vocab_path
        self.save_path = g_config.save_path                                 # 模型训练结果
        self.log_path = g_config.log_path
        self.embedding_pretrained = torch.tensor(pkl.load(open(self.emb_path, 'rb')).astype('float32'))
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
        self.num_filters = 250                                          # 卷积核数量(channels数)


'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
