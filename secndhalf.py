
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

