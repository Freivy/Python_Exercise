import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
from gensim.models import Word2Vec

class Deepwalk:
    def __init__(self, data_file):
        self.data_file = data_file
        self.graph = None
        self.embed_model = None

    def build_graph(self):
        """
        两种创造图的方式
        edges = pd.DataFrame({
            "source": [0, 1, 2, 0],
            "target": [2, 2, 3, 2],
            "my_edge_key": ["A", "B", "C", "D"],
            "weight": [3, 4, 5, 6],
            "color": ["red", "blue", "blue", "blue"],
        })
        """

        # G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),
        # nodetype=None,data=[('weight',int)])
        edges = pd.read_csv(self.data_file)
        graph = nx.from_pandas_edgelist(edges, edge_attr=["weight", "color"],create_using=nx.MultiGraph())
        self.graph = graph
        return graph

    def get_randomwalk(self, node, path_length):
        random_walk = [node]
        cur = node
        for i in range(path_length - 1):
            cur_nbrs = list(self.graph.neighbors(cur))
            # 去除已在ls中的节点
            cur_nbrs = list(set(cur_nbrs) - set(random_walk))
            if len(cur_nbrs) == 0:
                break
            # 随机挑选一个
            cur = random.choice(cur_nbrs)
            random_walk.append(cur)
        return random_walk

    def gen_sample(self):
        all_nodes = list(self.graph.nodes)
        random_walks = []
        for node in tqdm(all_nodes):
            # 每个节点随机走5次，2088个节点最后生成10440个长度为10的序列
            for i in range(5):
                random_walks.append(self.get_randomwalk(node, 10))
        return random_walks

    def emb_model(self, embed_size=128, window_size=5, workers=3, iter_num=5, **kwargs):
        # 训练embeding
        # Word2Vec(movie_list, size=10, window=5, sg=1, hs=0, min_count=1)
        sentences = self.gen_sample()
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers  # 训练并行数
        kwargs["window"] = window_size
        kwargs["iter"] = iter_num  # 随机梯度下降迭代次数
        model = Word2Vec(**kwargs)
        self.embed_model = model
        return model

    def get_embedding(self):
        if self.embed_model == {}:
            print('model not train yet')
            return {}
        embeddings = {}
        for word in self.graph.nodes():
            embeddings[word] = self.embed_model.wv[word]
        return embeddings








