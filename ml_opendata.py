from numpy import *
import operator

def file2matrix(filename):
	# parse tab separeted data to list
	# and label
	fp = open(filename)
	data = fp.readlines()
	dataLen = len(data)
	dataSet = zeros((dataLen, 3))
	i = 0
	classLabel = []
	for record in data:
		recordCol = record.strip().split("\t")
		dataSet[i, :] = recordCol[0:3]
		classLabel.append(recordCol[-1])
		i += 1
	return dataSet, classLabel

def autonorm(dataSet):
	# normalize data
	# (data - min)/(max-min)
	dataRow = dataSet.shape[0]
	dataMin = dataSet.min(axis=0)
	dataMax = dataSet.max(axis=0)
	dataRange = dataMax - dataMin
	dataNorm = (dataSet - dataMin)/dataRange
	return dataNorm, dataMin, dataRange

def classify0(dataIn, dataSet, dataLabel, k):
	# hitung jarak antara dataIn dengan masing2 dataSet
	dataDist = ((dataIn - dataSet) ** 2).sum(axis=1) ** 0.5
	# sortir jarak dari yg pendek ke yg panjang
	distIdx  = dataDist.argsort()
	# ambil k label dari jarak yg paling pendek
	# dari k jarak paling pendek ambil nilai label yg paling banyak	
	labelCount = {}
	for i in range(k):
		curLabel = dataLabel[distIdx[i]]
		labelCount[curLabel] = labelCount.get(curLabel,0) + 1
	sortedLabel = sorted(labelCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedLabel[0][0]

def testClassifier():
	# ambil 10% data  dari dataSet
	# test data tersebut dgn classifier
	# cocokkan hasil classifier dengan label
	# apabila tidak cocok tambahkan count error
	pass

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

test , tost  = createDataSet()
