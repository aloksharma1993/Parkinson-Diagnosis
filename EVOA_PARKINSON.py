import random
import math
import sys
import time

from sklearn import svm , metrics ,datasets, neighbors, linear_model
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

# variables

Mf = 19
Mv = 15
MAXITER = 20
SWITCHPROB = 0.4
TH = 10
trainX = []
trainY = []
testX = []
testY = []
vult = []
fitness = []
gBest = -1
best = -1
bbb = -1
# input dataset
def readData() :
	global trainX,trainY,testX,testY
	f = open('gi/train_inp.txt','rb')
	for line in f :
		trainX.append( [float(i) for i in line.split()])
	f = open('gi/train_out.txt','rb')
	for i in f :
		trainY.append(int(i))
	f = open('gi/test_inp.txt','rb')
	for line in f :
		testX.append( [float(i) for i in line.split()] )
	f = open('gi/test_out.txt','rb')
	for i in f :
		testY.append(int(i))

	trainX = np.array(trainX)
	trainY = np.array(trainY)
	testX = np.array(testX)
	testY = np.array(testY)

	#normalization
	#trainX = preprocessing.normalize(trainX, norm='l2')
	#testX = preprocessing.normalize(testX, norm='l2')

	#scaling
	#min_max_scaler = preprocessing.MinMaxScaler()
	#trainX = min_max_scaler.fit_transform(trainX)
	#testX = min_max_scaler.fit_transform(testX)

	#print trainX[0]
#print trainX[0:5]


#print

def printf() :
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best
	#print trainX[0:5]
	#print len(trainX[0])
	#print trainY[0:5]
	#print len(trainY)
	#print testX[0:5]
	#print testY[0:5]
	print vult[best]
	print 'gbest %f ' % gBest


# initialize vultures

def init_sequence() :
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best
	for i in range(0,Mv) :
		x = random.randrange(1,Mf+1)
		v = []
		for i in range(0,x) :
			v.append(1)
		for i in range(x,Mf):
			v.append(0)

		random.shuffle(v);
		vult.append(v);

def calc_fitness(x) :
	#print 'calc fitness'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best
	trTrainX = trainX
	trTestX = testX

	trTrainX = np.array(trTrainX)
	trTestX = np.array(trTestX)

	#print trTrainX
	#print 'x',
	#print len(x)
	ct = 0
	lx = len(x)
	for i in range(0,lx) :
		if x[i] == 0 :
			#print trTrainX.shape
			#print trTestX.shape
			#print 'i-ct ',
			#print i-ct
			trTrainX = np.delete(trTrainX,i-ct,1)
			trTestX = np.delete(trTestX,i-ct,1)

			ct = ct + 1

	if trTrainX.shape[1] == 0 :
		return 0.0

	#SVM
	#clf = svm.SVC(gamma = 0.01,C = 100)
	#clf.fit(trTrainX,trainY)
	#predicted = np.array(clf.predict(trTestX))

	# KNN
	clf = KNeighborsClassifier()
	#clf.fit(trTrainX,trainY)
	#predicted = clf.predict(trTestX)

	#Decision Tree Classifier
	#clf = DecisionTreeClassifier()
	#clf.fit(trTrainX,trainY)
	#predicted = clf.predict(trTestX)

	#Logistic Regression
	#clf = LogisticRegression()
	#clf.fit(trTrainX,trainY)
	#predicted = clf.predict(trTestX)

	# Naive Bayes
	#clf = GaussianNB()
	clf.fit(trTrainX,trainY)
	predicted = clf.predict(trTestX)

	return f1_score(testY,predicted, average='binary')

def init_fitness() :
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	for i in range(0,Mv):
		fitness.append(calc_fitness(vult[i]))
		if gBest < fitness[i] :
			gBest = fitness[i]
			best = i

	bbb = max(bbb,gBest)
	#print fitness
	#print gBest

def pebble_tossing() :
	#print 'pebble tossing'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	for i in range(0,Mv) :
		switch = random.random()

		if switch < SWITCHPROB :
			#print 'local'
			pos = random.randrange(0,Mf);
			pebble = vult[i][pos]

			nnvult = vult[i][:pos] + vult[i][pos+1:]

			smin = random.randrange(0,Mf-1)
			smax = random.randrange(0,Mf-1)
			if smax < smin :
				smin,smax = smax,smin

			while smax - smin + 1 > TH :
				smax = random.randrange(0,Mf-1)
				if smax < smin :
					smin,smax = smax,smin

			#print smin
			#print smax
			#print nnvult
			#print len(nnvult)
			d = -1
			for j in range(smin,smax+1) :
				vvult = nnvult[:];
				#print vvult
				#print nnvult
				vvult.insert(j,pebble)
				#print j
				#print pebble
				#print vvult
				#print nnvult
				#print len(vvult)
				y = calc_fitness(vvult)
				if y > d:
					d = y
					pos = j

			nnvult.insert(pos,pebble)
			if fitness[i] < d :
				fitness[i] = d
				vult[i] = nnvult[:]

		else :
			#print 'global'
			'''for j in range(0,5):
				target = random.randrange(0,Mf)
				nextTarget = 0 if target+1 >= Mf else target+1
				A = vult[best][target]
				B = vult[best][nextTarget]
				posa ,posb = -1,-1
				nvult = vult[i][:]
				nvult[target] = A
				nvult[nextTarget] = B
				fitn = calc_fitness(nvult)
				if fitn > fitness[i] :
					fitness[i] = fitn
					vult[i] = nvult[:]
			'''
			smin = random.randrange(0,Mf-1)
			smax = random.randrange(0,Mf-1)
			if smax < smin :
				smin,smax = smax,smin

			while smax - smin + 1 > TH :
				smax = random.randrange(0,Mf-1)
				if smax < smin :
					smin,smax = smax,smin

			nvult = vult[i][:]

			for j in range(smin,smax+1) :
					nvult[j] = vult[best][j]

			fitn = calc_fitness(nvult)
			bbb = max(bbb,fitn)
			if fitn > fitness[i] :
				fitness[i] = fitn
				vult[i] = nvult[:]




def rolling_twigs() :
	#print 'rolling twig'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	for i in range(0,Mv) :
		lmax = random.randrange(0,Mf)
		lmin = random.randrange(0,Mf)
		if lmax < lmin :
			lmin,lmax = lmax,lmin

		window_len = lmax - lmin + 1

		ds = random.randrange(0,window_len)
		dr = random.randrange(0,2)

		aa = lmin if lmax - ds +1 > lmax else lmax - ds + 1
		bb = lmin if lmin + ds > lmax else lmin + ds

		st = aa if dr == 0 else bb
		arr = []

		for j in range(0,window_len) :
			arr.append(vult[i][st])
			st = lmin if st + 1 > lmax else st+1

		nvult = vult[i][:]
		for j in range(0,window_len) :
			nvult[j+lmin] = arr[j]

		nfit = calc_fitness(nvult)
		bbb = max(bbb,nfit)
		if fitness[i] < nfit :
			fitness[i] = nfit
			vult[i] = nvult[:]

def change_Of_Angle() :
	#print 'change of angle'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	for i in range(0,Mv) :
		lmax = random.randrange(0,Mf)
		lmin = random.randrange(0,Mf)
		if lmax < lmin :
			lmin,lmax = lmax,lmin

		nvult = vult[i]
		nvult[lmin:lmax+1] = reversed(nvult[lmin:lmax+1])
		nfit = calc_fitness(nvult)
		bbb = max(bbb,nfit)
		if fitness[i] < nfit :
			fitness[i] = nfit
			vult[i] = nvult[:]


def egyptian_vulture() :
	#print 'egyptian vulure'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	readData()
	init_sequence()
	init_fitness()

	for i in range(0,MAXITER) :
		print 'iteration number : %d ' % i
		pvult = vult[:]
		pfitness = fitness[:]

		pebble_tossing()
		rolling_twigs()
		change_Of_Angle()

		for j in range(0,Mv) :
			if fitness[j] < pfitness[j] :
				fitness[j] = pfitness[j]
				vult[j] = pvult[j][:]
			if gBest < fitness[j] :
				gBest = fitness[j]
				best = j
			bbb = max(bbb,gBest)

	print 'GBEST %f ' % gBest
	print 'BBB %f ' % bbb

# main

def main():
	start_time = time.time()
	egyptian_vulture()
	printf()
	print("--- %s seconds ---" % (time.time() - start_time))
	return 0

if __name__ == '__main__':
	main()

