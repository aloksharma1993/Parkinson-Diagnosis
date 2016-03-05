import random
import math
import time


import numpy as np
from sklearn import svm , metrics ,datasets, neighbors, linear_model
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing


#variables
MAX_ITER = 100
POP_SIZE = 49
MAX_FEATURE = 19
V_MAX = 6
V_MIN = 0

trainX = []
trainY = []
testX =  []
testY =  []

GBest = []     # postion string of best particle
gbest = 0    # best fitness value -> Gbest's pbest's fitness
c1 = 2         #Cognitive learning factor
c2 = 2         #Social Learning factor
Particles = []


def sigmoid(velocity):
	  x = math.exp(-velocity)
	  return (1 / (1 + x))
	  

class Particle:
	def __init__(self):
		f = random.randint(1,MAX_FEATURE) # number of features present in THE particle
		self.pos = []
		for i in range(0, f):
			self.pos.append(1)
		for i in range(f,MAX_FEATURE):
			self.pos.append(0)
		random.shuffle(self.pos)
		#print pos
		self.vel = []
		for i in range(MAX_FEATURE):    #vel = array([random() for i in range(MAX_FEATURE)])
			self.vel.append(random.uniform(0,V_MAX))
 		self.pbest = self.pos
		self.pbestFit = 0.0    #fitness value of paricle at it's pbest
		self.fitness = 0.0
	
	def get_velocity():
		return vel
	def get_pos():
		return pos
	def get_pbest():    # pbest bit string
		return pbest
	def get_pbestFit(): # fitness value of pbest solution
		return pbestFit	

#input dataset
def readData():
	global trainX, trainY, testX, testY
	f = open('gi/train_inp.txt','rb')
	for line in f:
		trainX.append([float(i) for i in line.split()])
	f = open('gi/train_out.txt','rb')
	for i in f:
		trainY.append(int(i))
	f = open('gi/test_inp.txt','rb')
	for line in f:
		testX.append([float(i) for i in line.split()])
	f = open('gi/test_out.txt','rb')
	for i in f:
		testY.append(int(i))
	trainX = np.array(trainX)
	trainY = np.array(trainY)
	testX = np.array(testX)
	tetsY = np.array(testY)

	
#initialize particles

def init_Particles():
	for i in range(POP_SIZE):
		p = Particle()
		#print p.pos
		Particles.append(p)  #The population array is created Hurray!


def calc_Fitness(x):
	global trainX , trainY, testX, testY
	trTrainX = trainX
	trTestX = testX
	
	trTrainX = np.array(trTrainX)
	trTestX = np.array(trTestX)
	
	ct = 0
	lx = len(x)
	for i in range(lx):
		if x[i] == 0:
			trTrainX = np.delete(trTrainX, i-ct,1)
			trTestX = np.delete(trTestX, i-ct, 1)
			ct += 1
	if trTrainX.shape[1] == 0:   # if number of columns == 0
		return 0.0
	clf = KNeighborsClassifier()
	
	clf.fit(trTrainX,trainY)
	
	predicted = clf.predict(trTestX)
		
	ans =  f1_score(testY, predicted, average = 'binary') 
	#print 'ans = %f' % ans*100
	#p.fitness = ans*100
	return ans*100


def update_velocity(p):   # update velocity of a particle
	v = p.vel
	posi = p.pos
	vnew = []             #updated velocity
	pb = p.pbest  #cognitive : particle's best position so far
	gb = GBest            #social: best postion in swarm
	for i in range(MAX_FEATURE):
		x = v[i] + c1 * random.random() * (pb[i] - posi[i]) \
				 + c2 * random.random() * (gb[i] - posi[i])
		vnew.append(x)
	p.vel = vnew   #updated velocity


def update_position(p):
	for d in range(MAX_FEATURE):
		if(random.random() < sigmoid(p.vel[d])):
			 p.pos[d] = 1
		else:
			 p.pos[d] = 0


def Binary_PSO():
	global GBest, gbest,MAX_ITER, Particles
	readData()
	init_Particles()
	#initializing the GBest as paricle 0' best
	GBest = Particles[0].pbest
	for i in range(MAX_ITER):
		start_time = time.time()
		for p in Particles:
			p.fitness = calc_Fitness(p.pos)
			if(p.fitness > p.pbestFit):
				p.pbest = p.pos
				p.pbestFit = p.fitness
			if(p.fitness > gbest):
				gbest = p.fitness
				GBest = p.pos
		for p in Particles:
			update_velocity(p)
			update_position(p)
		print ('time in one iteration = %s ' % (time.time() - start_time))
		print 'GBEST fitness value is %f' % gbest	
	#print 'Best fitness value %d, with feature string, %s' gbest, GBest  
	print  'GBEST fitness value %f'  % gbest
	print  'GBEST feature set %s' % GBest

def main():
	start_time = time.time()
	Binary_PSO()
	print("****** %s second ******" % (time.time() - start_time))
	return 0

if __name__ == '__main__':
	main()

	

				
	


		
	
	
	
	
		  

  
