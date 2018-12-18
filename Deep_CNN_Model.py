
from Layers import *


class DeepConvNet:
    '''
    网络结构如下所示
            conv - relu - conv- relu - pool -
            conv - relu - conv- relu - pool -
            conv - relu - conv- relu - pool -
            affine - relu - dropout - affine - dropout - softmax

    '''
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):
        
        # 权重初始化 前一层的几个神经元有连接
        pre_node_nums = np.array(
            [1* 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size])

        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU的情况下的H初始化
        self.params = {}
        # 通道数
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                       pre_channel_num,
                                                                                       conv_param['filter_size'],
                                                                                       conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num'] 
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 生成层===========
        self.model = []
        
        # 卷积层 1
        self.model.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.model.append(Relu())
        # 卷积层 2
        self.model.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.model.append(Relu())
        self.model.append(Pooling(pool_h=2, pool_w=2, stride=2))
        # 卷积层 3
        self.model.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.model.append(Relu())
        # 卷积层 4
        self.model.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.model.append(Relu())
        self.model.append(Pooling(pool_h=2, pool_w=2, stride=2))
        # 卷积层 5
        self.model.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.model.append(Relu())
        #卷积层 6

        self.model.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.model.append(Relu())
        self.model.append(Pooling(pool_h=2, pool_w=2, stride=2))
        # 全连接层
        self.model.append(Affine(self.params['W7'], self.params['b7']))
        self.model.append(Relu())

        self.model.append(Dropout(0.5))
        self.model.append(Affine(self.params['W8'], self.params['b8']))
        self.model.append(Dropout(0.5))
        # 输出层
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.model:
            # 判断是否是Dropout层
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x


    # 计算损失函数
    def loss(self, x, t):
        y = self.predict(x, train_flg=True)

        # 返回最后一层
        return self.last_layer.forward(y, t)

    # 求准确率
    def accuracy(self, x, t, batch_size=100):

        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        # 正确数 /  样本数
        return acc / x.shape[0]

    def gradient(self, x, t):

        # forward

        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.model.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i + 1)] = self.model[layer_idx].dW
            grads['b' + str(i + 1)] = self.model[layer_idx].db

        return grads



