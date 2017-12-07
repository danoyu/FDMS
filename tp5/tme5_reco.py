import numpy as np
from sklearn import manifold, datasets
from matplotlib import pyplot as plt
from  time import  time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import pandas as pd


# pass in column names for each CSV and read them using pandas.
# Column names available in the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

# print users.shape
# print (users.head())

# print ratings.shape
# print (ratings.head())
#
# print items.shape
# print items.head()


# t0 = time()
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, metric='precomputed ')
# Y = tsne.fit_transform(users)
# plt.scatter(Y[:, 0], Y[:, 1],  cmap=plt.cm.Spectral)
# plt.show()
# t1 = time()
# # print("t-SNE: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(2, 5, 10,  projection='3d')
# plt.scatter(Y[:, 0], Y[:, 1],Y[:, 2],  c=color, cmap=plt.cm.Spectral)
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')


#dict { user : dict {film : rating} }
def loadMovieLens(path='ml-100k'):

  # Get movie titles
  movies={}
  for line in open(path+'/u.item'):
    (id,title)=line.split('|')[0:2]
    movies[id]=title

  # Load data
  prefs={}
  for line in open(path+'/u.data'):
    (user,movieid,rating,ts)=line.split('\t')
    prefs.setdefault(user,{})
    prefs[user][movies[movieid]]=float(rating)
  return prefs


def read_lines(filename):
    return list(row[0:3] for row in list(csv.reader( open(filename, 'rb'), delimiter='\t')) )


def user_indexed(data):
    user_index = {}
    for [user_id, item_id, rating] in data:
        if user_id not in user_index:
            user_index[user_id] = {item_id: rating}
        else:
            user_index[user_id][item_id] = rating
    return user_index


def item_indexed(data):
    item_index = {}
    for [user_id, item_id, rating] in data:
        if item_id not in item_index:
            item_index[item_id] = {user_id: rating}
        else:
            item_index[item_id][user_id] = rating
    return item_index


pref = loadMovieLens()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, metric='precomputed ')
Y = tsne.fit_transform(pref)
plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
plt.show()
