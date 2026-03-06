import torch
import torch.nn as nn

class LargeFeatureExtractor(torch.nn.Sequential):
    """
    LargeFeatureExtractorは、入力次元から指定された出力次元までの特徴を抽出するための多層パーセプトロン（MLP）です。
    
    Attributes:
        input_dim (int): 入力の特徴量の次元数
        output_dim (int): 出力の特徴量の次元数
        hidden_dims (list[int]): 隠れ層の次元数。デフォルトは[100, 50, 10]。
    """

    def __init__(self, input_dim, output_dim):
        """
        LargeFeatureExtractorのコンストラクタ。
        
        Args:
            input_dim (int): 入力データの次元数
            output_dim (int): 出力データの次元数
            hidden_dims (list[int], optional): 隠れ層の次元数。デフォルトは[100, 50, 10]。
        """
        super(LargeFeatureExtractor, self).__init__()

        # 最初の線形変換 (入力次元 -> 隠れ層1)
        self.add_module('linear1', torch.nn.Linear(input_dim, input_dim*10))
        # ReLU 活性化関数
        self.add_module('relu1', torch.nn.ReLU())

        # 2番目の線形変換 (隠れ層1 -> 隠れ層2)
        self.add_module('linear2', torch.nn.Linear(input_dim*10, input_dim*5))
        # ReLU 活性化関数
        self.add_module('relu2', torch.nn.ReLU())

        # 3番目の線形変換 (隠れ層2 -> 隠れ層3)
        self.add_module('linear3', torch.nn.Linear(input_dim*5, input_dim*2))
        # ReLU 活性化関数
        self.add_module('relu3', torch.nn.ReLU())

        # 最後の線形変換 (隠れ層3 -> 出力層)
        self.add_module('linear4', torch.nn.Linear(input_dim*2, output_dim))