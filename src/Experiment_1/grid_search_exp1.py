from __future__ import print_function
import re, glob, sys, time, os, random
from time import gmtime, strftime
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import warnings
from sklearn.model_selection import cross_val_predict, cross_val_score
from statistics import mean
from datetime import datetime
from time import time
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
warnings.filterwarnings("ignore", category=UserWarning)
import pprint
from random import shuffle




pipeline = Pipeline([
	('tfidf', TfidfVectorizer(analyzer='word')),
	('clf', LinearSVC())
])


parameters = {
	'tfidf__ngram_range': [(1,2),(1,1),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(3,4),(4,4)],
	#'tfidf__analyzer': ('word', 'char'),
	'tfidf__lowercase': (True, False),
	#'tfidf__max_df': (0.01, 1.0), # ignore words that occur as more than 1% of corpus
	'tfidf__min_df': (1, 2, 3), # we need to see a word at least (once, twice, thrice)
	'tfidf__use_idf': (False, True),
	#'tfidf__sublinear_tf': (False, True),
	'tfidf__binary': (False, True),
	'tfidf__norm': (None, 'l1', 'l2'),
	#'tfidf__max_features': (None, 2000, 5000),
	'tfidf__stop_words': (None, 'english'),

	#'tfidfchar_ngram_range': ((1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,2),(2,3),(2,4),(2,5),(2,6),(3,3),(3,4),(3,5),(3,6),(4,4),(4,5),(4,6),(5,5),(5,6),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7)),
	
	
	'clf__C':(0.1, 1, 5)
}


def balance(Xtrain,Ytrain):
	#v = Ytrain.count('violation')
   # nv = Ytrain.count('non-violation')
	#print(v, nv)
	print('more balancing')
	v = [i for i,val in enumerate(Ytrain) if val=='violation']
	nv = [i for i,val in enumerate(Ytrain) if val=='non-violation']
	if len(nv) < len(v):
		v = v[:len(nv)]
		Xtrain = [Xtrain[j] for j in v] + [Xtrain[i] for i in nv]
		Ytrain = [Ytrain[j] for j in v] + [Ytrain[i] for i in nv]
	if len(nv) > len(v):
		nv = nv[:len(v)]
		Xtrain = [Xtrain[j] for j in v] + [Xtrain[i] for i in nv]
		Ytrain = [Ytrain[j] for j in v] + [Ytrain[i] for i in nv]
	
	#print(Ytrain.count('violation'),Ytrain.count('non-violation'))
	#print('LEN', len(Xtrain), len(Ytrain))
	return Xtrain, Ytrain
	
	

def extract_text(starts, ends, cases, violation):
	facts = []
	D = []
	years = []
	for case in cases:
		contline = ''
		year = 0
		with open(case, 'r') as f:
			for line in f:
				#print(line)
				dat = re.search('^([0-9]{1,2}\s\w+\s([0-9]{4}))', line)
				if dat != None:
					#print('Yaay')
					#date = dat.group(1)
					year = int(dat.group(2))
					break
			if year>0:
				years.append(year)
				#print(year)
				wr = 0
				for line in f:
					if wr == 0:
						if re.search(starts, line) != None:
							wr = 1
					if wr == 1 and re.search(ends, line) == None:
						contline += line
						contline += '\n'
					elif re.search(ends, line) != None:
						break
				facts.append(contline)
	for i in range(len(facts)):
		D.append((facts[i], violation, years[i])) 
	return D

def extract_parts(article, violation, part, path):
	cases = glob.glob(path)
	#print(cases)

		
	facts = []
	D = []
	years = []
	
	if part == 'relevant_law':
		for case in cases:
			year = 0
			contline = ''
			with open(case, 'r') as f:
				for line in f:
					dat = re.search('^([0-9]{1,2}\s\w+\s([0-9]{4}))', line)
					if dat != None:
						 #date = dat.group(1)
						year = int(dat.group(2))
						break
				if year> 0:
					years.append(year)
					wr = 0
					for line in f:
						if wr == 0:
							if re.search('RELEVANT', line) != None:
								wr = 1
						if wr == 1 and re.search('THE LAW', line) == None and re.search('PROCEEDINGS', line) == None:
							contline += line
							contline += '\n'
						elif re.search('THE LAW', line) != None or re.search('PROCEEDINGS', line) != None:
							break
					facts.append(contline)
		for i in range(len(facts)):
			D.append((facts[i], violation, years[i]))
		
	if part == 'facts':
		starts = 'THE FACTS'
		ends ='THE LAW'
		D = extract_text(starts, ends, cases, violation)
	if part == 'circumstances':
		starts = 'CIRCUMSTANCES'
		ends ='RELEVANT'
		D = extract_text(starts, ends, cases, violation)
	if part == 'procedure':
		starts = 'PROCEDURE'
		ends ='THE FACTS'
		D = extract_text(starts, ends, cases, violation)
	if part == 'procedure+facts':
		starts = 'PROCEDURE'
		ends ='THE LAW'
		D = extract_text(starts, ends, cases, violation)
	return D


def run_pipeline(part):
	v = extract_parts(article, 'violation', part, '/data/p282832/HUDOC/train/'+article+'/violation/*.txt')
	nv = extract_parts(article, 'non-violation', part, '/data/p282832/HUDOC/train/'+article+'/non-violation/*.txt')
	
	test_v = extract_parts(article, 'violation', part, '/data/p282832/HUDOC/test_violations/'+article+'/*.txt')
	
	#v = extract_parts(article, 'violation', part, './train/'+article+'/violation/*.txt')
	#nv = extract_parts(article, 'non-violation', part, './train/'+article+'/non-violation/*.txt')
	
	#test_v = extract_parts(article, 'violation', part, './test_violations/'+article+'/*.txt')
	
	#print(len(v), len(nv), len(test_v))
	#trainset = v + nv + test_v

	print('balancing the number of cases...')
	if len(nv) < len(v):
		v = v[:len(nv)]
	if len(nv) > len(v):
		nv = nv[:len(v)]
	trainset = v+nv+test_v

	Xtrain = [i[0] for i in trainset]
	Ytrain = [i[1] for i in trainset]
	#print(Ytrain.count('violation'), Ytrain.count('non-violation'))
	YearTrain = [i[2] for i in trainset]
	
	Xtest2 = []
	Ytest2 = []
	Xtest1 = []
	Ytest1 = []
	X = []
	Y = []
	
	for i in range(len(YearTrain)):
		if YearTrain[i] >= 2016:
			Xtest2.append(Xtrain[i])
			Ytest2.append(Ytrain[i])
		elif YearTrain[i] == 2014 or YearTrain[i] == 2015:
			Xtest1.append(Xtrain[i])
			Ytest1.append(Ytrain[i])
		else:
			X.append(Xtrain[i])
			Y.append(Ytrain[i])
	
	#print('X,Y', Y.count('violation'), Y.count('non-violation'))
	
	X, Y = balance(X, Y)
	#print('X,Y balanced', Y.count('violation'), Y.count('non-violation'))
	Xtrain = X
	Ytrain = Y
	
	#print('2014-2015', Ytest1.count('violation'), Ytest1.count('non-violation'))
	#print('2016-2016', Ytest2.count('violation'), Ytest2.count('non-violation'))
	Xtest1, Ytest1 = balance(Xtest1, Ytest1)
	#print('2014-2015 balanced', Ytest1.count('violation'), Ytest1.count('non-violation'))
	Xtest2, Ytest2 = balance(Xtest2, Ytest2)
	#print('2014-2016 balanced', Ytest2.count('violation'), Ytest2.count('non-violation'))
	X_1 = X + Xtest1
	Y_1 = Y + Ytest1
	
	X_3 = []
	Y_3 = []
	
	d_whole = {}
	for i in range(len(Xtest1)):
		d_whole[Xtest1[i]] = Ytest1[1]
	for i in range(len(Xtest2)):
		d_whole[Xtest2[i]] = Ytest2[1]
		
	for key, value in d_whole:
		X_3.append(key)
		Y_3.append(value)
		
	X_3 = X_3[:len(Xtest1)]
	Y_3 = Y_3[:len(Xtest1)]
		

		

	print('Training on', str(Ytrain.count('violation') + Ytrain.count('non-violation')), 'cases', '\nCases for testing on (2014-2015):', Ytest1.count('violation')+Ytest1.count('non-violation'), '\nCases for testing on (2016-2017):', Ytest2.count('violation')+Ytest2.count('non-violation'))
	
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=24, verbose=1)
	t0 = time()
	grid_search.fit(Xtrain, Ytrain)

	print("done in %0.3fs" % (time() - t0))
	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))
		
	
	k = sorted(parameters.keys())
	vec = TfidfVectorizer()
	clf = LinearSVC()
	pipeline_test = Pipeline([('tfidf', vec), ('clf', clf)])
	pipeline_test.set_params(**best_parameters)
	print('fitting the best model')
	pipeline_test.fit(Xtrain, Ytrain)
	
	Ypredict = cross_val_predict(pipeline_test, Xtrain, Ytrain, cv=3)
	print('Accuracy:', accuracy_score(Ytrain, Ypredict) )
	print('\nClassification report:\n', classification_report(Ytrain, Ypredict))
	print('\nConfusion matrix:\n', confusion_matrix(Ytrain, Ypredict), '\n\n_______________________\n\n')
	accuracies.append(accuracy_score(Ytrain, Ypredict))
	
	print('testing on 2014-2015')
	Ypredict = pipeline_test.predict(Xtest1)
	print('Accuracy:', accuracy_score(Ytest1, Ypredict) )
	print('\nClassification report:\n', classification_report(Ytest1, Ypredict))
	print('\nConfusion matrix:\n', confusion_matrix(Ytest1, Ypredict), '\n\n_______________________\n\n')
	
	print('testing on 2016-2017')
	Ypredict = pipeline_test.predict(Xtest2)
	print('Accuracy:', accuracy_score(Ytest2, Ypredict) )
	print('\nClassification report:\n', classification_report(Ytest2, Ypredict))
	print('\nConfusion matrix:\n', confusion_matrix(Ytest2, Ypredict), '\n\n_______________________\n\n')
	
	print('testing on 2014-2017 (whole period)')
	Ypredict = pipeline_test.predict(X_3)
	print('Accuracy:', accuracy_score(Y_3, Ypredict) )
	print('\nClassification report:\n', classification_report(Y_3, Ypredict))
	print('\nConfusion matrix:\n', confusion_matrix(Y_3, Ypredict), '\n\n_______________________\n\n')

	print('testing on 2016-2017 with more training data')
	pipeline_test.fit(X_1, Y_1)
	Ypredict = pipeline_test.predict(Xtest2)
	print('Accuracy:', accuracy_score(Ytest2, Ypredict) )
	print('\nClassification report:\n', classification_report(Ytest2, Ypredict))
	print('\nConfusion matrix:\n', confusion_matrix(Ytest2, Ypredict), '\n\n_______________________\n\n')
	
	print('train on training set + 2016-2017 & test')
	X_2 = X+Xtest2
	Y_2 = Y+Ytest2
	pipeline_test.fit(X_2,Y_2)
	Ypredict = pipeline_test.predict(Xtest1)
	print('Accuracy:', accuracy_score(Ytest1, Ypredict) )
	print('\nClassification report:\n', classification_report(Ytest1, Ypredict))
	print('\nConfusion matrix:\n', confusion_matrix(Ytest1, Ypredict), '\n\n_______________________\n\n')

if __name__ == "__main__":
	split = 0.8
	article = sys.argv[1]
	parts = ['facts', 'circumstances', 'relevant_law', 'procedure', 'procedure+facts']
	accuracies = []
	current_time = strftime("%H:%M:%S", gmtime())
	sys.stdout = open('time_results/'+ article +'_time.txt', 'w')
	for part in parts:
		print('Training on', part)
		run_pipeline(part)
	print(parts)
	print(accuracies)
	sys.stdout.close()


	
