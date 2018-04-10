
from sklearn import metrics
import numpy as np
from numpy.linalg import inv
class fda(object):
	def __init__(self, path):
		skin = open(path,'r')
		rl = skin.readlines()
		#print(rl)

		src_data = []
		count = [0,0]
		for e in rl:
		    if(e.find('\n') != -1):
		        e = e[:-1]
		    a= e.split(' ')
		    a = [float(i) for i in a]
		    src_data.append(np.array(a))

		skin.close()
		#print(src_data)


		self.train_1 = np.array([e[:-1] for e in src_data[:216] if e[-1] == 1])
		self.train_2 = np.array([e[:-1] for e in src_data[:216] if e[-1] == 2])

		self.test_1 = np.array([e[:-1] for e in src_data[216:] if e[-1] == 1])
		self.test_2 = np.array([e[:-1] for e in src_data[216:] if e[-1] == 2])


	def scatt(self,x):
	    mean = np.mean(x, axis = 0)
	    sct = x - mean
	    Sct = np.dot(sct.T,sct)
	    #print(Sct.shape)
	    return Sct
	def get_weights(self,cl_1, cl_2):
	    Sw = self.scatt(cl_1) + self.scatt(cl_2)
	    #print('Sw shape: ', Sw.shape)
	    diff = np.mean(cl_1, axis = 0) - np.mean(cl_2, axis = 0)
	    #print('Mean diff shape: ', diff.shape)
	    W = np.dot( inv(Sw) , diff )
	    #print('Weights shape: ', W.shape)
	    return W
	def project(self, W, cl_1, cl_2):
	    Y_1 = np.dot(W.T,cl_1.T)
	    Y_2 = np.dot(W.T,cl_2.T)
	    return Y_1, Y_2

	def find_threshold(self,y1,y2):
	    mean1, mean2 = np.mean(y1), np.mean(y2)
	    min_val = min(min(y1),min(y2))
	    max_val = max(max(y1),max(y2))
	    threshold = min_val
	    y = np.concatenate((y1,y2),axis=0)
	    best = 0
	    while(threshold < max_val):
	        correct = 0
	        for e in y1:
	            if(e >= threshold):
	                correct+=1.0
	        #print('Class 1 Acc', correct / (y1.shape[0]))
	        
	        for e in y2:
	            if(e < threshold):
	                correct+=1.0
	        acc = correct / (y1.shape[0] + y2.shape[0])
	        
	        #print('Acc: ', acc)
	        if(acc>best):
	            best = acc
	            #print(best)
	            best_th = threshold
	        threshold+=0.00001
	    return best_th,best

	def eval(self,y1,y2,threshold):
	    correct = 0
	    for e in y1:
	        if(e >= threshold):
	            correct+=1.0
	    
	    for e in y2:
	        if(e < threshold):
	            correct+=1.0
	    
	    acc = correct / (y1.shape[0] + y2.shape[0])
	    return acc


	def predict(self,y1,y2,threshold):  
	    preds = []
	    for e in y1:
	        if(e >= threshold):
	            preds.append(0.0)
	        else:
	            preds.append(1.0)
	    
	    for e in y2:
	        if(e >= threshold):
	            preds.append(0.0)
	        else:
	            preds.append(1.0)
	        
	        
	    return np.array(preds)

	def build(self):
    	#Get Weights
		self.W = self.get_weights(self.train_1,self.train_2)
		#Evaluate Train
		Y_1, Y_2 = self.project(self.W, self.train_1, self.train_2)
		self.threshold, self.best_train_acc= self.find_threshold(Y_1,Y_2)

	def evaluate(self):
		
		self.ytest1, self.ytest2 = self.project(self.W,self.test_1,self.test_2)
		self.test_acc = self.eval(self.ytest1, self.ytest2, self.threshold)

		return self.test_acc

	def ROC(self):
		preds = self.predict(self.ytest1,self.ytest2,self.threshold)
		y_test = np.concatenate(( np.zeros((len(self.test_1))), np.ones( (len(self.test_2)) ) ), axis=0)

		fpr, tpr, th = metrics.roc_curve(y_test, preds)
		roc_auc = metrics.auc(fpr, tpr)
		return fpr,tpr,th, roc_auc
