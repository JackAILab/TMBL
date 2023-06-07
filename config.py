import argparse
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import os

# path to a pretrained word embedding file
# word_emb_path = ''
data_str = "mosei" # mosi or mosei or ur_funny
word_emb_path = '/data/ProjectData/Multimodal/MOSEI/embedding_and_mapping.pt' # 改为mosi测试的时候，这里记得也要改 MOSI MOSEI UR_FUNNY
assert(word_emb_path is not None)

# project_dir = Path(__file__).resolve().parent.parent # 默认设置为当前文件路名的上一层，再上一层目录
# sdk_dir = project_dir.joinpath('/home/jack/Project/MutiModal/SentimentAnalysis/PSMixer/CMU_MultimodalSDK')
# data_dir = project_dir.joinpath('datasets')  
# data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
#     'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}

project_dir = "/data/ProjectData/"
sdk_dir ='/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/MultimodalSA/CMU_MultimodalSDK'
data_dir = os.path.join(project_dir, 'Multimodal') # /data/ProjectData/Multimodal
data_dict = {'mosi': os.path.join(data_dir,'MOSI'), 'mosei': os.path.join(data_dir,
     'MOSEI'), 'ur_funny': os.path.join(data_dir,'UR_FUNNY')}

optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh,"gelu":nn.GELU}


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)


        self.dataset_dir = data_dict[self.data.lower()]
        self.sdk_dir = sdk_dir
        # Glove path
        self.word_emb_path = word_emb_path

        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")  # saved-model's name
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--milestones', type=list, default=[10, 30, 80])
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--trials', type=int, default=3)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')

    parser.add_argument('--rnncell', type=str, default='lstm')  # lstm or GRU
    parser.add_argument('--embedding_size', type=int, default=300)  # embedding size in bert
    parser.add_argument('--hidden_size', type=int, default=256)  # modality embedding size
    parser.add_argument('--mlp_hidden_size', type=int, default=64)  # mlp-communicator hidden size
    parser.add_argument('--dropout', type=float, default=0.5) # 0.2 - 0.5 adjust
    parser.add_argument('--depth', type=int, default=1)  # mlp-communicator depth number

    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')

    # three loss weights
    # parser.add_argument('--cls_weight', type=float, default=1)
    # parser.add_argument('--polar_weight', type=float, default=0.1)
    # parser.add_argument('--scale_weight', type=float, default=0.1)

    parser.add_argument('--model', type=str,
                        default='AOTN', help='one of {AOTN, }')

    # parser.add_argument('--model', type=str,
    #                     default='PS_Mixer', help='one of {PS_Mixer, }')

    parser.add_argument('--test_duration', type=int, default=1)

    # Data
    parser.add_argument('--data', type=str, default=data_str)  # mosi or mosei or ur_funny

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    print(kwargs.data)
    if kwargs.data == "mosi":
        kwargs.num_classes = 1 
        kwargs.batch_size = 64 # PXmixer 128 18G 左右显存 %%% Swim 64 9G 左右显存
        kwargs.depth = 1
    elif kwargs.data == "mosei":
        kwargs.num_classes = 1
        kwargs.batch_size = 48 # PXmixer 64 20G 左右显存 %%% Swim 32 24G 左右显存
        kwargs.depth = 1
    elif kwargs.data == "ur_funny":
        kwargs.num_classes = 1 # Jack Change 2 to 1
        kwargs.batch_size = 32
    else:
        print("No dataset mentioned")
        exit()

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)