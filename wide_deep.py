import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import dataloader
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralMF.")
    parser.add_argument('--path', nargs='?', default='/dataset/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ratings.csv',
                        help='Choose a dataset.')
    parser.add_argument('--layers', nargs='+', default=[64,32,16,8],
                        help='num of layers and nodes of each layer. embedding size is (2/1st layer) ')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--gmf_regs', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--mlp_regs', nargs='+', default=[0,0,0,0],
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.(1 or 0)')
    parser.add_argument('--patience', type=int, default=10,
                        help='earlystopping patience')
    parser.add_argument('--pretrain_gmf', nargs='?', default='',
                        help='')
    parser.add_argument('--pretrain_mlp', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='pretrain trade off between GMF:alpha || MLP:1-alpha ')
    return parser.parse_args()


class wide_deep():
    def __init__(self, X_wide,y_wide,X_deep,y_deep):
        wide_input = keras.layers.Input(shape = X_wide.shape[1],
                                        name="wide_input")




        deep_input = keras.layers.Input(shape = X_deep.shape[1],
                                        name="deep_input")






if __name__ == "__main__":
    args = parse_args()
    layers = args.layers
    num_factors = args.num_factors
    mlp_regs = args.mlp_regs
    gmf_regs = args.gmf_regs
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience
    pretrain_gmf = args.pretrain_gmf
    pretrain_mlp = args.pretrain_mlp
    alpha = args.alpha

    loader = dataloader.dataloader()


