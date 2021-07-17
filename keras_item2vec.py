from keras.models import Model
import random
import numpy as np
import collections

def skipgram_reader_generator(corpus, window =2, neg_size):
    vocab = []
    for ls in corpus:
        vocab.extend(ls)
    counter = collections.Counter(vocab)
    sample_weights = [counter[w]**0.75 for w in counter]

    def reader():
        # 每次读取一行
        for ls_id in range(len(corpus)):
            tmp_ls = corpus[ls_id]
            for i in range(len(tmp_ls)):
                context_list = []
                target = tmp_ls[i]
                j = i - window  #前后各采一个window
                while j <= i + window and j < len(tmp_ls):
                    if j >= 0 and j != i:
                        context_list.append(tmp_ls[j])
                        yield ((target, tmp_ls[j]), 1)
                    j += 1
                # 负采样
                for _ in range(len(context_list)):
                    ne_idx = random.choice(vocab, sample_weights, k=neg_size)
                    # 当负采样在上下文list内时重新抽
                    while ne_idx in context_list:
                        ne_idx = random.choice(vocab, sample_weights, k=neg_size)





