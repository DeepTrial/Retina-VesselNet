
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""


class ModelBase(object):
    """
    模型基类
    """

    def __init__(self, config):
        self.config = config  # 配置
        self.model = None  # 模型

    def save(self):
        """
        存储checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Saving model...")
        json_string = self.model.to_json()
        open(self.config.hdf5_path+self.config.exp_name + '_architecture.json', 'w').write(json_string)
        print("[INFO] Model saved")

    def load(self, checkpoint_path):
        """
        加载checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Loading model checkpoint ...\n")
        self.model.load_weights(self.config.hdf5_path+self.config.exp_name+ '_best_weights.h5')
        print("[INFO] Model loaded")

    def build_model(self):
        """
        构建模型
        """
        raise NotImplementedError
