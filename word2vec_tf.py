from gensim.models import Word2Vec
import os
import pandas as pd

class skipgram_item:
    def __init__(self, input_file):
        self.model = self.get_train_data(input_file)

    def get_train_data(self, input_file):
        if not os.path.exists(input_file):
            return.0
        # 对于较低分的样本直接删除
        score_thr = 4.0
        ratingsDF = pd.read_csv(input_file, index_col=None, sep='::',
                                header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
        ratingsDF = ratingsDF[ratingsDF['rating'] > score_thr]
        ratingsDF['movie_id'] = ratingsDF['movie_id'].apply(str)
        movie_list = ratingsDF.groupby('user_id')['movie_id'].apply(list).values
        print('training...')
        model = Word2Vec(movie_list, size=10, window=5, sg=1, hs=0, min_count=1)
        return model

    def recommend(self, movieid, k):
        movieID = str(movieid)
        rank = self.model.most_similar(movieID, topn=k)
        return rank

if __name__ == '__main__':

    datapath = '/Users/fudi/Documents/fd_project/movielens/ml-1m'
    #moviepath = datapath+'/movies.csv'
    moviepath = datapath+'/ratings.dat'
    userpath = datapath+'/users.dat'

    movie_emb = skipgram_item(moviepath)
    top10 = movie_emb.recommend(2, 10)
    print('recommend', top10)


