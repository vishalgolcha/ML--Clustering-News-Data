import numpy as np 
import pandas as pd 
import nltk 
import pymongo
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# from collections import counter
from named_entity import get_more_entities
from named_entity import get_keywords
from named_entity import get_entity 
import string 
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib as mpl
import matplotlib.pyplot as plt

client =MongoClient('mongodb://digi1:digi1234@52.21.107.21:27017')
db=client.links 
collection=db.toi_feed

# titles =["Trump","Hillary","Bill"]
titles=[]
data=[]   

# str1 = "Donald John Trump is an American businessman , television personality , author and politician"
# str2 = "Hillary Diane Rodham Clinton is an American politician and the presumptive nominee of the Democratic Party for President of the United States in the 2016 election"
# str3 = "Bill Maher is an American comedian, writer , producer , political commentator , actor , media critic , and television host "

for x in collection.find():
    # titles.append(x['title'])
    # d=get_keywords(x['title'])
    # b=[]
    # for i in range(len(d)):
    #     if i<3:
    #         b.append(d[i])
    # delimiter=' '
    # k=delimiter.join(b)
    titles.append('')
    # for i in d :

    if x['text']== '':
        data.append(x['title'])
    else :
        data.append(x['text'])

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc])
    return lexicon

# def tf(term, document):
#   return freq(term, document)




def freq(term, document,num):
  # return document.count(term)
    cnt =0
    mulfact=1 
    # print len(document)
    for i in range(len(document)):
        if term in entity_collection[num]:
            mulfact=3
        if term==document[i] :
            cnt+=1

    return cnt*mulfact


def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]

def numDocsContaining(word, doclist):
    doccount = 0
    for j in range(len(doclist)):
        if freq(word,doclist[j],j) > 0:
            doccount +=1
    return doccount 

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    # print 1+np.log(n_samples / float(1+df))
    return 1+np.log(n_samples / float(1+df))

def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat



stop = stopwords.words('english')
# print stop 

						

# save all text as list here 
# str= "abc.\nabc.\nabc"	

pdata=[x.replace("\n","") for x in data] 

entity_collection=[ list(set(get_more_entities(x)+get_keywords(x))) for x in pdata ]
# print "entity"
# print entity_collection
# print "collection"

pdata=[ x.replace("."," ") for x in data ]
# str=str.replace
# print str
#replace the full stops with spaces 


doc_count={}

tdata=[ nltk.word_tokenize(x) for x in pdata ]
# print tdata

for i in tdata :
    # print i 
    for j in i :
        # print j 
        if j in stop:
            i.remove(j)
# tdata=[ i.remove(j) for i in tdata for j in i if j in stop ]
# print tdata 

vocabulary = build_lexicon(tdata)
# print "vocab"
# print vocabulary
# print "bingo"
doc_term_matrix = []
# print tf.items()

# print freq('Mumbai',tdata[0],0)
for k in range(len(tdata)):
    # print doc
    # print 'The doc is "' + doc + '"'
    tf_vector = [ freq(word,tdata[k],k) for word in vocabulary]
    # print tf_vector
    # tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    # print 'The tf vector for Document %d is [%s]' % ((mydoclist.index(doc)+1), tf_vector_string)
    doc_term_matrix.append(tf_vector)
    
    
# print 'All combined, here is our master document term matrix: '
# print doc_term_matrix

doc_term_matrix_l2 = []
for vec in doc_term_matrix:
    doc_term_matrix_l2.append(l2_normalizer(vec))

# print 'A regular old document term matrix: ' 
# print np.matrix(doc_term_matrix)
# print '\nA document term matrix with row-wise L2 norms of 1:'
# print np.matrix(doc_term_matrix_l2)

my_idf_vector = [idf(word, tdata) for word in vocabulary]

# print 'Our vocabulary vector is [' + ', '.join(list(vocabulary)) + ']'
# print 'The inverse document frequency vector is [' + ', '.join(format(freq, 'f') for freq in my_idf_vector) + ']'

my_idf_matrix = build_idf_matrix(my_idf_vector)

doc_term_matrix_tfidf = []

#performing tf-idf matrix multiplication
for tf_vector in doc_term_matrix:
    doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))

#normalizing
doc_term_matrix_tfidf_l2 = []
for tf_vector in doc_term_matrix_tfidf:
    doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))
                                    
# print vocabulary
# print np.matrix(doc_term_matrix_tfidf_l2) 
pca = PCA( n_components=2 )
# print pca.fit_transform(np.matrix(doc_term_matrix_tfidf_l2))
red_matrix=pca.fit_transform(np.matrix(doc_term_matrix_tfidf_l2))
dist = 1 - cosine_similarity(red_matrix)

num_clusters = 15
km = KMeans(n_clusters=num_clusters,n_init=30)
km.fit(dist)
clusters = km.labels_.tolist()
print clusters
 


pos=red_matrix
xs, ys = pos[:, 0], pos[:, 1]
#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e',5:'#800000',6:'#808080',7:'#FFC300',
                  8:'#7d5147', 9:'#FF5733',10:'#05de40',11:'#34495e',12:'#f9e79f',13:'#a569bd',14:'#9c640c' }

#set up cluster names using a dict
cluster_names = {0: '0', 
                 1: '1', 
                 2: '2', 
                 3: '3', 
                 4: '4',
                 5: '5',
                 6:'6',
                 7:'7',
                 8:'8',
                 9:'9',
                 10:'10',
                 11:'11',
                 12:'12',
                 13:'13',
                 14:'14'
                 }

# matplotlib inline
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(25, 16)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

# grouped = frame['rank'].groupby(frame['cluster'])

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=6)
plt.savefig('start4.png', dpi=400)
plt.show()

# print km
# clusters=km.labels_
# print clusters 

# h = .02    

# x_min, x_max = dist[:, 0].min() - 1, dist[:, 0].max() + 1
# y_min, y_max = dist[:, 1].min() - 1, dist[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
# print type(Z)
# Z = Z.reshape(xx.shape)

# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')

# plt.plot(dist[:, 0],dist[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()