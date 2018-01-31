import logging
import mxnet as mx

from util import *

logging.getLogger().setLevel(logging.DEBUG)

batch_size = 1000

train_x, train_y, test_x, test_y = load_single_train_data('.cache/dba/data', 1)

mnist = mx.test_utils.get_mnist()

train_iter = mx.io.NDArrayIter(train_x.reshape(train_x.shape[0], 1, 16, 8), train_y, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(test_x.reshape(test_x.shape[0], 1, 16, 8), test_y, batch_size)

# train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
# val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(3,3), num_filter=64)
relu1 = mx.sym.Activation(data=conv1, act_type="relu")
# pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=relu1, kernel=(3,3), num_filter=64)
relu2 = mx.sym.Activation(data=conv2, act_type="relu")
# pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=relu2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
relu3 = mx.sym.Activation(data=fc1, act_type="relu")
# second fullc
fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=512)
relu4 = mx.sym.Activation(data=fc2, act_type="relu")

fc3 = mx.sym.FullyConnected(data=relu4, num_hidden=128)
relu5 = mx.sym.Activation(data=fc3, act_type="relu")

fc4 = mx.sym.FullyConnected(data=relu5, num_hidden=8)

# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

# create a trainable module on GPU 0
lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu())
# train with the same
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
                num_epoch=28)
