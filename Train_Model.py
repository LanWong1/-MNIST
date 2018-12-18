import sys , os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gzip
from Deep_CNN_Model import DeepConvNet
from Train import Trainer
# from mnist import load_mnist

files = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__)) # 当前目录
pickle_file = dataset_dir + "/mnist.pkl"

# 转换为one-hot
def change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

# 读取数据
def load_mnist(normalize=True, flatten=True, one_hot_label=False):

    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    # 正规化
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    #one_hot
    if one_hot_label:
        dataset['train_label'] = change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = change_one_hot_label(dataset['test_label'])
    # 展开
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])



'''
开始训练
'''
# 读取数据
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)

#print(x_test.shape)


# 创建模型
network = DeepConvNet()

# 创建训练对象
trainer = Trainer(network, x_train, y_train, x_test, y_test,
                  epochs=20,mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
# 开始训练
trainer.train()



