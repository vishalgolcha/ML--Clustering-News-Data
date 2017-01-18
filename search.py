import nltk 
import pymongo
from pymongo import MongoClient
import string 
import math
from nltk import ngrams
import urllib

client =MongoClient('mongodb://digi1:digi1234@52.21.107.21:27017')
db=client.links 
collection=db['toi_feed_2']

a= raw_input().split(',')

# def ng():
t= a[0].split(' ')

links=[]

bigrams=[]
trigrams=[]

#get ngrams 
b = ngrams(a[0].split(),2)
f = [e for e in b]
f = [list(e) for e in f]
f = [' '.join(h) for h in f]
a.extend(f)

b = ngrams(a[0].split(),3)
f = [e for e in b]
f = [list(e) for e in f]
f = [' '.join(h) for h in f]
a.extend(f)

a.extend(f)
a=list(set(a))

cut=0 

#retrieve modi images from database depending on query

for x in collection.find({'VisualData': { '$exists': True }}):
	# print x
	# print "\n"
	# cut+=1
	for i in x['VisualData']:
		flag=0
		for j in a :
			if j in i['dtext']:
				flag=1
				break
		if flag==1:
			links.extend(i['vsrc'].split('<__>'))
			# print i['vsrc'].split('<__>')

print a 

for i in links:
	if i=='' or i==' ':
		links.remove(i)

print links 
cnt=0
for link in links:
	cnt+=1
	urllib.urlretrieve(link,"/home/ubuntu/imtag/modi/"+str(cnt)+link[-5:])

