# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

from torch import nn
import torch


class TextExtract(nn.Module):

    def __init__(self, opt):
        super(TextExtract, self).__init__()

        self.embedding_local = nn.Embedding(opt.vocab_size, 512, padding_idx=0)  # 生成大小为vocab_size（可以装这么多个词），维度为512的向量词典
        self.embedding_global = nn.Embedding(opt.vocab_size, 512, padding_idx=0)
        self.dropout = nn.Dropout(0.3)  # 以0.3的概率扔掉神经元防止过拟合
        self.lstm = nn.LSTM(512, 2048, num_layers=1, bidirectional=True, bias=False)  # 构建LSTM模型的网络

    def forward(self, caption_id, text_length):
        # 构建前向传播模型，每一步表示处理步骤
        text_embedding_global = self.embedding_global(caption_id)
        text_embedding_global = self.dropout(text_embedding_global)
        text_embedding_global = self.calculate_different_length_lstm(text_embedding_global, text_length, self.lstm)

        text_embedding_local = self.embedding_local(caption_id)
        text_embedding_local = self.dropout(text_embedding_local)
        text_embedding_local = self.calculate_different_length_lstm(text_embedding_local, text_length, self.lstm)

        return text_embedding_global, text_embedding_local

    def calculate_different_length_lstm(self, text_embedding, text_length, lstm):
        text_length = text_length.view(-1)  # 改变维度
        _, sort_index = torch.sort(text_length, dim=0, descending=True)  # dim=0按列排序，降序，返回两个值分别为排序后的数据和其在原矩阵中的下标
        _, unsort_index = sort_index.sort()#这个sort不懂

        sortlength_text_embedding = text_embedding[sort_index, :]#不懂
        sort_text_length = text_length[sort_index]
        # print(sort_text_length)
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(sortlength_text_embedding,
                                                                  sort_text_length,
                                                                  batch_first=True)#

        # self.lstm.flatten_parameters()
        packed_feature, _ = lstm(packed_text_embedding)  # [hn, cn]
        total_length = text_embedding.size(1)
        sort_feature = nn.utils.rnn.pad_packed_sequence(packed_feature,
                                                        batch_first=True,
                                                        total_length=total_length)  # including[feature, length]

        unsort_feature = sort_feature[0][unsort_index, :]
        unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
                          + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2

        return unsort_feature.permute(0, 2, 1).contiguous().unsqueeze(3)
