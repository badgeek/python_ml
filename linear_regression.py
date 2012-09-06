import matplotlib.pyplot as plt
import numpy as np


def file2matrix(filename):
	txt = np.loadtxt(fname=filename, delimiter=",")	
	data = txt[:,0]
	label = txt[:,1]
	return data, label


def hypothesis(feature, weight):
	m = feature.shape[0]
	w = np.matrix(weight).transpose()
	x = np.matrix(np.ones(shape=[m,2]))
	for i in range(m):
		x[i,1] = feature[i]
	return x * w

def error(feature, weight, label):
	h = hypothesis(feature,weight)
	l = np.mat(label).transpose()
	e = h - l
	return e

def gradientDescent(feature, label):
	m = feature.shape[0]
	a = 0.01
	iter = 1500
	j0 = 0
	j1 = 0
	for i in range(iter):
		w = [j0,j1]
		e = error(feature, w, label)
		j0 = j0 - (a/m *  e.sum())
		j1 = j1 - (a/m *  np.multiply(e, np.matrix(feature).transpose()).sum())
	print "j0:", j0, " j1:",j1
