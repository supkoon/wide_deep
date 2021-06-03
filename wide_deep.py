import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import dataloader
import argparse
import matplotlib.pyplot as plt
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralMF.")
    parser.add_argument('--path', nargs='?', default='./datasets/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens',
                        help='Choose a dataset.')
    parser.add_argument('--layers', nargs='+', default=[1024,512,256],
                        help='num of layers and nodes of each layer ')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='test_size.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--deep_regs', nargs='+', default=[0,0,0],
                        help="Regularization for deep")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.(1 or 0)')
    parser.add_argument('--patience', type=int, default=10,
                        help='earlystopping patience')

    return parser.parse_args()


class wide_deep(keras.Model):
    def __init__(self, layers = [1024,512,256], deep_regs= [0,0,0],**kwargs):
        super().__init__(**kwargs)
        self.num_layers=  len(layers)
        self.n_neuron = list(map(int,layers))
        self.deep_regs = list(map(float, deep_regs))
        self.layers_list = []
        for index in range(self.num_layers):
            layer = keras.layers.Dense(self.n_neuron[index], kernel_regularizer=keras.regularizers.l2(self.deep_regs[index]),
                                       activation=keras.activations.relu,
                                       name=f'layer{index}')
            self.layers_list.append(layer)
        self.wide_deep_output = keras.layers.Dense(1, activation='sigmoid', name='wide_deep_output')

    def call(self,inputs):

        wide_input, deep_input = inputs
        deep_result = deep_input
        for layer in self.layers_list:
            deep_result = layer(deep_result)

        concat_wide_deep = keras.layers.concatenate([wide_input,deep_result])

        wide_deep_output = self.wide_deep_output(concat_wide_deep)

        return  wide_deep_output


if __name__ == "__main__":
    args = parse_args()
    layers = args.layers
    deep_regs = args.deep_regs
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience
    test_size = args.test_size


    #wide inputs,deep inputs
    loader = dataloader.dataloader(args.path + args.dataset)
    X_wide_train,X_deep_train,X_wide_test,X_deep_test,y_train,y_test = loader.make_binary_set(test_size=test_size)

    print(X_wide_train.shape)
    print(X_deep_train.shape)

    model = wide_deep(layers = layers, deep_regs=deep_regs)

    if learner.lower() == "adagrad":
        model.compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate), loss= keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.BinaryAccuracy()])
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate), lloss= keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.BinaryAccuracy()])
    elif learner.lower() == "adam":
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss= keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.BinaryAccuracy()])
    else:
        model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss= keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.BinaryAccuracy()])

    early_stopping_callback = keras.callbacks.EarlyStopping(patience=patience,restore_best_weights=True)
    model_out_file = 'wide_deep%s.h5' % (datetime.now().strftime('%Y-%m-%d-%h-%m-%s'))
    model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)

    if args.out:
        history = model.fit([X_wide_train, X_deep_train], y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=([X_wide_test, X_deep_test], y_test), callbacks=[early_stopping_callback,
                                                                                             model_check_cb]
                            )
    else:
        history = model.fit([X_wide_train, X_deep_train], y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=([X_wide_test, X_deep_test], y_test), callbacks=[early_stopping_callback]
                            )


    pd.DataFrame(history.history).plot()
    plt.show()

