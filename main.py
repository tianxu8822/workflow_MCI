from utils import read_json
from model_wrapper import Experiment_wrapper
import torch
import multiprocessing
import time
import pynvml
import os
import sys
torch.backends.benchmark = True


def cnn_main(gpu_index_tmp):
    path = os.getcwd().replace('\\', '/')
    # print(path)

    # path = path[:-len(path.split('/')[-1])]
    path = path + '/'
    # print(111, path, os.getcwd())

    names = []
    if 'result.csv' in os.listdir(path):
        os.remove('{}result.csv'.format(path))
    # exit()
    for file_ in os.listdir(path):
        if file_.endswith('.csv'):
            names.append(file_[:-4])
    # print(path, names)
    # exit()
    with torch.cuda.device(gpu_index_tmp):
        for name in names:
            cnn_setting = config['cnn']
            seed = config["seed"]
            cnn = Experiment_wrapper(fil_num=cnn_setting['fil_num'],
                                     drop_rate=cnn_setting['drop_rate'],
                                     batch_size=cnn_setting['batch_size'],
                                     categories=['CN', 'MCI'],
                                     data=name,
                                     data_dir=cnn_setting['data_dir'].replace('example', name),
                                     learn_rate=cnn_setting['learning_rate'],
                                     train_epoch=cnn_setting['train_epochs'],
                                     seed=seed,
                                     metric='ACC',
                                     gpu=gpu_index_tmp)
            cnn.external_verification()


def get_gpus_info(gpus_tmp):
    gpus_info_tmp = dict()
    pynvml.nvmlInit()
    for item in gpus_tmp:
        handle = pynvml.nvmlDeviceGetHandleByIndex(item)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus_info_tmp[item] = meminfo.free/1024**2
    return gpus_info_tmp


if __name__ == "__main__":
    print("MCI identification starting ...")

    old_path = os.getcwd()
    os.chdir(sys.path[0])

    gpus = [0]

    # to perform CNN training #####################################
    config = read_json('./config.json')
    gpus_info = get_gpus_info(gpus)
    print(gpus_info)
    cnn_main(gpus[0])
    os.chdir(old_path)
