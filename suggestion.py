import nltk
from nltk import ngrams
import json
import pymongo
from pymongo import MongoClient
from nltk.corpus import stopwords




client =MongoClient('mongodb://digi1:digi1234@52.21.107.21:27017')
db=client.links 
collection=db.toi_feed
# word="gistai"

stop = stopwords.words('english')
m=[')','(','we','he','she']
stop.extend(m)

# tags=db.cumulative

words=[]
sent=""
cnt =0 
for x in collection.find():
	for i in range(len(x['VisualData'])):
		cnt+=1
		# (x['VisualData'][i]['dtext'])
		words.extend(set(nltk.word_tokenize(x['VisualData'][i]['dtext'])))
		# print '\n'
# print cnt
red=0
# filt=["AT","WP","PRP"]

for i in words :
	# print nltk.pos_tag(i) 
	if i in stop :
		red+=1
		words.remove(i)
# print words
# print red

#

mix=(list(set(words)))
lo= [x.lower() for x in mix]  

tup=enumerate(list(set(lo)))
tup = [reversed(x) for x in tup]
keyval=dict(tup)
# print keyval

losos=[]

docwords={}

# for t in words:
# 	for x in collection.find():	
# 		for i in range(len(x['VisualData'])):	
# 			if  x['VisualData'][i]['dtext'].find(t) != -1 :
# 				docwords[t]+=1
# [ for t in words for x in collection.find() for i in range(len(x['VisualData'])) if t in x['VisualData'][i]['dtext']]				

def idf(word,n_doc):
    return 1+np.log( n_doc/float(1+docwords[word]))

for x in collection.find():	
	for i in range(len(x['VisualData'])):	
		sos=x['VisualData'][i]['dtext']
		sos=nltk.word_tokenize(sos)
		for k in sos:
			docwords[k]+=1
		sos=[keyval[d.lower()] for d in sos if d not in stop]
		
		# print sos
		# print '\n'
	losos.append(list(set(sos)).sort())

# calculate idf 
s=raw_input()
k=s.split(' ')
# take input 
# get all documents which have the term and simultaneously score them  




