import statistics
from sklearn import svm, metrics
from svmutil import *
import re
import string
import numpy
import pandas as ps
from sklearn.metrics import confusion_matrix



def parseInputData(path):
	lines = open(path,'r')

	trainingBitWords = list()
	trainingLabels = list()
	for line in lines:
		completeTrainFile = (re.split(r'\t*', str(line)))
		if len(completeTrainFile) > 2:
			trainingBitWords.append(re.split(r'im*', str(completeTrainFile[1])))
			trainingLabels.append(str(completeTrainFile[2]))
	

	trainingBits = list()

	for item in range(len(trainingBitWords)):
		trainingBits.append(list(trainingBitWords[item][1]))

	for i in range(len(trainingBits)):
		for j in range(128):
			trainingBits[i][j] = int(trainingBits[i][j])

	for i in range(len(trainingLabels)):
			trainingLabels[i] = ord(trainingLabels[i])-97	#ascii value


	lines.close()

	return trainingBits, trainingLabels

def svmLearn(path):

	i=j=x=y=0
	trainingPoints, trainingLabels = parseInputData(path)

	'''print(len(trainingPoints))
				x= int(len(trainingPoints)*0.8)
				x1= int(len(trainingPoints)*0.2)
				y1= int(len(trainingLabels)*0.8)
				y= int(len(trainingLabels)*0.2)
			
				print("x, x1 and y1, y",x,x1,y1,y)
				
				trbits=[[0 for j in range(128)] for i in range(x)]
				tebits= [[0 for j in range(128)] for i in range(y)]
				trlabels= list()
				telabels= list()
			
				for i in range(0 , (x)): #4300
					for j in range(128):
						trbits[i][j] = trainingPoints[i][j]
			
				for i in range(0,y):	#1075
					for j in range(128):
						tebits[i][j] = trainingPoints[i+x][j]
			
				for i in range(0 , x):
					trlabels.append(trainingLabels[i])
			
				for i in range(0,y):
					telabels.append(trainingLabels[i+x])'''

	#Training
	model = svm_train(trainingLabels,trainingPoints, '-c 0.1 -t 0')  

	'''#Training Accuracy
				t_label,t_acc,t_val=svm_predict(trlabels,trbits,model)
				t_ACC, t_MSE, t_SCC = evaluations(trlabels, t_label)
			
				#Validation Accuracy
				p_label, p_acc, p_val = svm_predict(telabels, tebits, model)
				v_ACC, v_MSE, v_SCC = evaluations(telabels, p_label)
			
				print("\n\nTraining")
				print (t_ACC)
				ACCTraining.append(t_ACC)
			
				print("\n\nValidation")
				print (v_ACC)
				ACCValidating.append(v_ACC)
			
				del trbits[:]
				del trbits[:]
				del trlabels[:]
				del telabels[:]'''

	return model

def testModelAccuracy(path, model):
	trainingPoints, trainingLabels = parseInputData(path)
	
	test_label, test_acc, test_val = svm_predict(trainingLabels, trainingPoints, model)
	test_ACC, test_MSE, test_SCC = evaluations(trainingLabels, test_label)
	
	Mat = confusion_matrix(trainingLabels, test_label)
	global TotalMat
	TotalMat = TotalMat+Mat
	
	ACCTesting.append(test_ACC)
	return TotalMat
	

def main():
	
	
	print('------------------------------Fold 0-------------------------------')
	modelf0 = svmLearn('ocr_fold0_sm_train.txt')
	print('------------------------------Fold 0-------------------------------')
	ConMat=testModelAccuracy('ocr_fold0_sm_test.txt', modelf0)

	print('------------------------------Fold 1------------------------------')
	modelf1 = svmLearn('ocr_fold1_sm_train.txt')
	print('------------------------------Fold 1-------------------------------')
	ConMat= testModelAccuracy('ocr_fold1_sm_test.txt', modelf1)
		
	print('------------------------------Fold 2------------------------------')
	modelf2 = svmLearn('ocr_fold2_sm_train.txt')
	print('------------------------------Fold 2-------------------------------')
	ConMat= testModelAccuracy('ocr_fold2_sm_test.txt', modelf2)
			
	print('------------------------------Fold 3------------------------------')
	modelf3 = svmLearn('ocr_fold3_sm_train.txt')
	print('------------------------------Fold 3-------------------------------')
	ConMat= testModelAccuracy('ocr_fold3_sm_test.txt', modelf3)
	
	print('------------------------------Fold 4------------------------------')
	modelf4 = svmLearn('ocr_fold4_sm_train.txt')
	print('------------------------------Fold 4-------------------------------')
	ConMat= testModelAccuracy('ocr_fold4_sm_test.txt', modelf4)
			
	print('------------------------------Fold 5------------------------------')
	modelf5 = svmLearn('ocr_fold5_sm_train.txt')
	print('------------------------------Fold 5-------------------------------')
	ConMat= testModelAccuracy('ocr_fold5_sm_test.txt', modelf5)
	
	print('------------------------------Fold 6------------------------------')
	modelf6 = svmLearn('ocr_fold6_sm_train.txt')
	print('------------------------------Fold 6-------------------------------')
	ConMat= testModelAccuracy('ocr_fold6_sm_test.txt', modelf6)
			
	print('------------------------------Fold 7------------------------------')
	modelf7 = svmLearn('ocr_fold7_sm_train.txt')
	print('------------------------------Fold 7-------------------------------')
	ConMat= testModelAccuracy('ocr_fold7_sm_test.txt', modelf7)
			
	print('------------------------------Fold 8------------------------------')
	modelf8 = svmLearn('ocr_fold8_sm_train.txt')
	print('------------------------------Fold 8-------------------------------')
	ConMat= testModelAccuracy('ocr_fold8_sm_test.txt', modelf8)

	print('------------------------------Fold 9------------------------------')
	modelf9 = svmLearn('ocr_fold9_sm_train.txt')
	print('------------------------------Fold 9-------------------------------')
	ConMat= testModelAccuracy('ocr_fold9_sm_test.txt', modelf9)

	for i in range(26):
		for j in range(26):
			ConMat[i][j]=ConMat[i][j]/10

	print (ConMat)



ACCTraining = list()
ACCValidating = list()
ACCTesting = list()
TotalMat = [[0 for j in range(26)] for i in range(26)]
main()
print("Accuracy ",ACCTesting)
'''print("Acc Training" , ACCTraining)
print("Acc Validating" , ACCValidating)
print("Acc Testing " , ACCTesting)
'''
