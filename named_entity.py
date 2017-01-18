import nltk
import json

#queries dbpedia for additonal entities
def get_more_entities(sample):
	a=get_keywords(sample)
	b=get_entity(sample)
	c=list(set(a)^set(b))
	more=[]
	for i in range(len(c)):
		try :
			annotations = spotlight.annotate('http://spotlight.sztaki.hu:2222/rest/annotate',\
				c[i],confidence=0.4,support=20,spotter='Default')
			# print annotations
			b.append(c[i])
		except :
			pass

	return list(set(b))



def get_keywords(sample):
	sentences = nltk.sent_tokenize(sample)
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
	# print tokenized_sentences 
	tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
	# print tagged_sentences
	filt=['NN','NNP','NNS','FW']
	d= [extra[i][0] for extra in tagged_sentences for i in range(len(extra)) if extra[i][1] in filt ]
	return list(set(d))

#detect named entities 
def get_entity(sample):
	sentences = nltk.sent_tokenize(sample)
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
	# print tokenized_sentences 
	tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
	# print tagged_sentences
	chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
	# print chunked_sentences

	# tags=['NN','NNP','NNS','FW']
	# d= [extra[i][0] for extra in tagged_sentences for i in range(len(extra)) if extra[i][1] in tags ]
	# keywords.extend(list(set(d)))
	def extract_entity_names(t):
	    entity_names = []

	    if hasattr(t, 'label') and t.label:
	        if t.label()=='NE':
	            entity_names.append(' '.join([child[0] for child in t]))
	        else:
	            for child in t:
	                entity_names.extend(extract_entity_names(child))

	    return entity_names

	entity_names = []

	for tree in chunked_sentences:
	    # Print results per sentence
	    # print extract_entity_names(tree)
	    entity_names.extend(extract_entity_names(tree))

	# Print all entity names
	# print entity_names

	# Print unique entity names
	x=set(entity_names)
	# for y in x 
	return list(x)