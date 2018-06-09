"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""


class TrainerBase(object):
    """
    训练器基类
    """

    def __init__(self, model, data, config):
        self.model = model  # 模型
        self.data = data  # 数据
        self.config = config  # 配置

    def train(self):
        """
        训练逻辑
        """
        raise NotImplementedError
