# coding=utf-8
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


#标准化，原数据非近似高斯分布的情况下，可能改变数据的分布


class DataGen:
    def __init__(self):
        self.dense_norm_feats = [...]  # 23个
        self.dense_feats = [...]  # 21个
        self.sparse_onehot_feats = [...]  # 8个
        self.sparse_feats = [...]  # 8个
        self.sparse_feat_dict = {
            'takeoffhour': 20,
            'arrivalhour': 21,
            'takeoffptime': 4,
            'sorttype': 6,
            'crafttypesize': 4,
            'lowprice_rn': 25,
            'highprice_rn': 160,
            'duration_rn': 153}

    def data_process(self):
        path = '/data/share/fd/list_rec0817.csv'
        data = pd.read_csv(path, '\t')
        data[self.dense_norm_feats] = (data[self.dense_norm_feats] - data[self.dense_norm_feats].mean()) / (
            data[self.dense_norm_feats].std())

        d = data[self.sparse_feats].copy()
        label_encoder = LabelEncoder()
        data[self.sparse_feats] = d.apply(label_encoder.fit_transform)

        with open('./label_encoder.pickle', 'wb') as f:
            pickle.dump(label_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)
        dataIn_dense_reshape = []
        dataIn_sparse_reshape = []
        dataOut_reshape = []
        with tqdm(data.groupby('transactionid')) as t:
            for key, group in t:
                dense_features = group[self.dense_feats +
                                       self.dense_norm_feats +
                                       self.sparse_onehot_feats].values
                sparse_features = group[self.sparse_feats].values
                if len(dense_features) > 25:
                    continue
                else:
                    dataIn_dense_reshape.append(dense_features)
                    dataIn_sparse_reshape.append(sparse_features)
                    dataOut_reshape.append(group[['isbuy']].values)
        np.save('./dataIn_dense_reshape.npy', dataIn_dense_reshape)
        np.save('./dataIn_sparse_reshape.npy', dataIn_sparse_reshape)
        np.save('./dataOut_reshape.npy', dataOut_reshape)
        t.close()

    def data_reload(self):
        self.data_process()
        dataIn_dense_reshape = np.load(
            './dataIn_dense_reshape.npy', allow_pickle=True)
        self.dataIn_dense_reshape = dataIn_dense_reshape.tolist()

        dataIn_sparse_reshape = np.load(
            './dataIn_sparse_reshape.npy', allow_pickle=True)
        self.dataIn_sparse_reshape = dataIn_sparse_reshape.tolist()

        dataOut_reshape = np.load('./dataOut_reshape.npy', allow_pickle=True)
        self.dataOut_reshape = dataOut_reshape.tolist()
        return dataIn_dense_reshape, dataIn_sparse_reshape, dataOut_reshape

    def data_padding(self):
        dataIn_dense_reshape, dataIn_sparse_reshape, dataOut_reshape = self.data_reload()
        max_len = max([len(x) for x in dataIn_dense_reshape])
        dataIn_dense = pad_sequences(
            dataIn_dense_reshape,
            maxlen=max_len,
            padding='post',
            value=0,
            dtype='float64')
        # shape =(num_sample, 25, 52) =(numsize,timestep,feat_dim)
        print('dense shape:', dataIn_dense.shape)
        dataIn_sparse = pad_sequences(
            dataIn_sparse_reshape,
            maxlen=max_len,
            padding='post',
            value=0,
            dtype='float64')
        # shape =( , 25, 8) =(numsize,timestep,feat_dim)
        print('sparse shape:', dataIn_sparse.shape)
        dataOut = pad_sequences(
            dataOut_reshape,
            maxlen=max_len,
            padding='post',
            value=0,
            dtype='float64')
        # shape =( , 25, 1) =(numsize,timestep,feat_dim)
        print('out shape:', dataOut.shape)
        train_in_dense, test_in_dense, train_in_sparse, test_in_sparse, train_data_out, test_data_out = train_test_split(
            dataIn_dense, dataIn_sparse, dataOut, test_size=0.3, random_state=1)
        return train_in_dense, test_in_dense, train_in_sparse, test_in_sparse, train_data_out, test_data_out
