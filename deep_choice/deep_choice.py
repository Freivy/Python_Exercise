# coding=utf-8
from sklearn import metrics
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
from tf.keras import initializers
from tf.keras import backend as K
from tf.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from Data_genrate import DataGen


class AttentionLayer(Layer):
    def __init__(self, att_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.att_dim = att_dim
        super(AttentionLayer, self).__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'supports_masking': self.supports_masking,
            'attention_dim': self.att_dim,
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3
        state_size = input_shape[-1]
        seq_len = input_shape[1]
        # attention_dim = hidden_dim,即encoder的隐藏层维度
        self.w2 = K.variable(self.init((state_size, self.att_dim)), name='w2')
        self.b = K.variable(self.init((self.att_dim,)), name='b')
        self.u = K.variable(self.init((self.att_dim, seq_len)), name='u')
        self.w_out = K.variable(self.init((state_size, 1)), name='w_out')
        # self.trainable_weights = [self.W, self.b, self.u]
        super(AttentionLayer, self).build(input_shape)

    # 为了传给下一层
    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        # size of x :[batch_size, seq_len, state_size]
        # d = tanh(W*en + b)
        '''
        uit = K.tanh(K.bias_add(K.dot(inputs, self.w2), self.b))
        ait = K.exp(K.squeeze(K.dot(uit, self.w1), -1)) #[sample_size,seq_len]
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = inputs * K.expand_dims(ait) [sample_size, seq_len, state_size]
        # output = K.sum(weighted_input, axis=1)

        return output
        '''
        en = inputs[:, -1, :]  # 取最后一步的hidden state [sample_size,1,state_size]
        d = K.tanh(
            K.bias_add(
                K.dot(
                    en,
                    self.w2),
                self.b))  # (sample_size,att_dim)
        ait = K.dot(d, self.u)  # [sample_size, seq_len]
        # ait
        x2 = K.dot(inputs, self.w_out)  # [sample_size,seq_len,1]

        if mask is not None:
            mask = K.cast(mask, K.floatx())  # [sample_size, seq_len]
            ait *= mask
        uj = K.exp(x2 * K.expand_dims(ait)) * K.expand_dims(mask)
        # weighted_input = K.exp(uj) *K.expand_dims(mask) #[200,seq_len,1]
        sum_weight = K.sum(uj, axis=1, keepdims=True) + 1e-32
        uj = uj / K.cast(sum_weight, K.floatx())
        return uj  # weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class DeepChoice():
    def __init__(self, sparse_val, sparse_feat_dict, dense_feats_all):
        # sparse feature
        self.sparse_val = sparse_val
        self.sparse_feat_dict = sparse_feat_dict

        # dense feature
        # dense_feats_all = dense_feats + dense_norm_feats + sparse_onehot_feats
        self.dense_inputs = Input(
            shape=(
                25,
                len(dense_feats_all)),
            name='dense')
        self.sparse_inputs, self.sparse_embed_kd = self.build_input()

    def build_input(self):
        sparse_inputs = []
        embeddings_kd = []
        for val in self.sparse_val:
            _input = Input(shape=(25, 1), name=val)
            voc_size = self.sparse_feat_dict[val]
            if voc_size < 10:
                emb_dim = int(voc_size // 2)
            else:
                emb_dim = 6 * int(pow(voc_size, 0.25))
            _embedk = Reshape([25,
                               emb_dim])(Embedding(voc_size + 1,
                                                   emb_dim,
                                                   embeddings_regularizer=tf.keras.regularizers.l2(0.7),
                                                   input_length=25)(_input))
            embeddings_kd.append(_embedk)  # 26个元素的list，每个元素是一个[None,8]的tensor
            sparse_inputs.append(_input)
        embeddings_kd = Concatenate(axis=-1)(embeddings_kd)
        return sparse_inputs, embeddings_kd  # [, 25,79]

    def build_model(self):
        mask_dense = tf.keras.layers.Masking(
            mask_value=0)(
            self.dense_inputs)  # 传递一个mask矩阵进去
        input_combined = Concatenate(
            axis=-1)([mask_dense, self.sparse_embed_kd])
        # unmasked_embedding = tf.cast(tf.tile(sentence_input, [1, 1, 1]), tf.float32)
        # sentence_input,mask = mask_array) #(, seq_len, 125)
        lstm_word = GRU(125, return_sequences=True)(input_combined)
        attn_word = AttentionLayer(64)(lstm_word)  # (none, 125*2),去掉了中间步的维度
        # preds = K.expand_dims(preds)

        model = Model(
            inputs=[
                self.dense_inputs,
                self.sparse_inputs],
            outputs=attn_word)
        return model


class MyCrossentropy(tf.keras.losses.Loss):
    # 基于mse自定义loss/loss=CustomMSE(0.4)
    def __init__(self, name="mask_crossentropy"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mask = y_pred._keras_mask  # K.any(K.not_equal(y_pred,0),axis =-1) #
        # mask = K.all(K.equal(y_true, mask_value), axis=-1)

        mask = K.cast(mask, K.floatx())  # [batch_size, seq_len]

        # multiply categorical_crossentropy with the mask
        y_pred = y_pred + 1e-32  # -sum(yi log(yi))
        loss = - K.cast(y_true, K.floatx()) * \
            tf.math.log(y_pred) * K.expand_dims(mask)
        # loss = K.categorical_crossentropy(y_true, y_pred) * mask #
        # cross_wntropy 返回batch_size个值 ，这里概率值不能为0，要先处理

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)  # / K.sum(mask)


def myauc(y_true, y_pred):
    mask = K.any(K.not_equal(y_pred, 0), axis=-1)  # y_pred._keras_mask
    # mask = K.all(K.equal(y_true, mask_value), axis=-1)
    mask = K.cast(mask, K.floatx())  # [batch_size, seq_len]

    # multiply categorical_crossentropy with the mask
    auc = tf.keras.metrics.AUC()
    # cross_wntropy 返回batch_size个值
    auc.update_state(y_true, y_pred, sample_weight=mask)

    return auc.result().numpy()


if __name__ == '__main__':
    data = DataGen()
    train_in_dense, test_in_dense, train_in_sparse, test_in_sparse, train_data_out, test_data_out = data.data_padding()
    dense_feats_all = data.dense_feats + data.dense_norm_feats + data.sparse_onehot_feats
    deep_choice = DeepChoice(data.sparse_feats, data.sparse_feat_dict, dense_feats_all)
    model = deep_choice.build_model()  # model.summary()
    model.compile(loss=MyCrossentropy(), optimizer='adam', metrics=['acc'], run_eagerly=True)  # 多分类并不适合用auc

    batch_size = 256
    early_stop = EarlyStopping(monitor='val_acc', patience=2, verbose=1, mode='auto')
    save_model = ModelCheckpoint('./model/deepchoice.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='auto')
    model_history = model.fit([train_in_dense, [val for val in train_in_sparse]], train_data_out, epochs=10,
                              batch_size=batch_size,
                              validation_data=([test_in_dense, [val for val in test_in_sparse]], test_data_out),
                              verbose=1, callbacks=[early_stop])
