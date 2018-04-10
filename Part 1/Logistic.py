
from sklearn import metrics
import numpy as np
from numpy.linalg import inv
class logistic(object):
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
			f = np.ones(15)
			for i in range(1,15):
			    f[i] = a[i-1]
			    f = [float(i) for i in f]
			src_data.append(np.array(f))

		skin.close()
		#print(src_data)
		x_points = [e[:-1] for e in src_data]
		y_points = [e[-1]-1 for e in src_data]

		self.x_train,self.y_train = np.array(x_points[:216]), y_points[0:216]
		self.x_test,self.y_test = np.array(x_points[216:]), y_points[216:]
	def sig(self,w):
		return (1 / (1 + np.exp(-1*w)))
	def p(self,x,B):
	    p_list = []
	    for e in x:
	        w = np.dot(e.T,B)
	        p_list.append(self.sig(w))
	    return np.array(p_list)
	def hess(self,x,B):
	    x_hs = []
	    for e in x:
	        w = np.dot(e.T,B)
	        a = np.dot(self.sig(w), (1-self.sig(w)))
	        a = np.dot(a,e.T)
	        x_hs.append(a)
	    return np.array(x_hs)
	def update(self,B, X, X_hs, Y, P):
	    P = self.p(X,B)
	    return B + np.dot( np.dot( inv( np.dot(X.T, X_hs) ), X.T ), (Y-P) )


	def newton(self,x_train,y_train):
	    
	    B = np.zeros(x_train[0].shape[0])
	    #print(B.shape)
	    X = x_train
	    Y = y_train
	    count = 0
	    while(count < 1000):
	        count+=1
	        P = self.p(X, B)
	        X_hs = self.hess(X,B)
	        B = self.update(B,X,X_hs,Y,P)
	        #print(B)
	    return B
	def eval(self,x,y,B):
	    correct = 0
	    #print(x)
	    for cn in range(x.shape[0]):
	        e = x[cn]
	        z = y[cn]
	        w = np.dot(e.T, B)
	        #print(sig(w), z)
	        if(self.sig(w) >= 0.5 and z == 1):
	            correct+=1.0
	        if(self.sig(w) < 0.5 and z == 0):
	            correct+=1.0
	    acc = correct / x.shape[0]
	    return acc

	def predict(self,x,B):
	    correct = 0
	    #print(x)
	    preds=  []
	    
	    for cn in range(x.shape[0]):
	        e = x[cn]
	        w = np.dot(e.T, B)
	        preds.append(self.sig(w))
	        '''if(sig(w) < 0.5):
	            preds.append(0.0)
	        else:
	            preds.append(1.0)'''
	            
	    return preds

	def build(self):
		self.B = self.newton(self.x_train,self.y_train)

	def evaluate(self):
		
		#train
		self.train_acc = self.eval(self.x_train,self.y_train,self.B)

		#train
		self.test_acc = self.eval(self.x_test,self.y_test,self.B)
		return self.train_acc, self.test_acc

	def ROC(self):
		preds = self.predict(self.x_test,self.B)

		fpr, tpr, th = metrics.roc_curve(self.y_test, preds)
		roc_auc = metrics.auc(fpr, tpr)
		return fpr,tpr,th, roc_auc
