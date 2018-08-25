import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class CollectData:
	def __init__(self):
		self.TP = []
		self.FP = []
		self.FN = []
		self.TN = []

	def reload(self,groundtruth,probgraph):
		"""
		:param groundtruth:  list,groundtruth image list
		:param probgraph:    list,prob image list
		:return:  None
		"""
		self.groundtruth = groundtruth
		self.probgraph = probgraph
		self.TP = []
		self.FP = []
		self.FN = []
		self.TN = []

	def statistics(self):
		"""
		calculate FPR TPR Precision Recall IoU
		:return: (FPR,TPR,AUC),(Precision,Recall,MAP),IoU
		"""
		for threshold in tqdm(range(0,255)):
			temp_TP=0.0
			temp_FP=0.0
			temp_FN=0.0
			temp_TN=0.0
			assert(len(self.groundtruth)==len(self.probgraph))

			for index in range(len(self.groundtruth)):
				gt_img=plt.imread(self.groundtruth[index])
				prob_img=plt.imread(self.probgraph[index])

				gt_img=(gt_img>0)*1
				prob_img=(prob_img>=threshold)*1

				temp_TP = temp_TP + (np.sum(prob_img * gt_img))
				temp_FP = temp_FP + np.sum(prob_img * ((1 - gt_img)))
				temp_FN = temp_FN + np.sum(((1 - prob_img)) * ((gt_img)))
				temp_TN = temp_TN + np.sum(((1 - prob_img)) * (1 - gt_img))

			self.TP.append(temp_TP)
			self.FP.append(temp_FP)
			self.FN.append(temp_FN)
			self.TN.append(temp_TN)

		self.TP = np.asarray(self.TP).astype('float32')
		self.FP = np.asarray(self.FP).astype('float32')
		self.FN = np.asarray(self.FN).astype('float32')
		self.TN = np.asarray(self.TN).astype('float32')

		FPR = (self.FP) / (self.FP + self.TN)
		TPR = (self.TP) / (self.TP + self.FN)
		AUC = np.round(np.sum((TPR[1:] + TPR[:-1]) * (FPR[:-1] - FPR[1:])) / 2., 4)

		Precision = (self.TP) / (self.TP + self.FP)
		Recall = self.TP / (self.TP + self.FN)
		MAP = np.round(np.sum((Precision[1:] + Precision[:-1]) * (Recall[:-1] - Recall[1:])) / 2.,4)

		iou=self.IoU()

		return (FPR,TPR,AUC),(Precision,Recall,MAP),iou

	def IoU(self,threshold=128):
		"""
		to calculate IoU
		:param threshold: numerical,a threshold for gray image to binary image
		:return:  IoU
		"""
		intersection=0.0
		union=0.0

		for index in range(len(self.groundtruth)):
			gt_img = plt.imread(self.groundtruth[index])
			prob_img = plt.imread(self.probgraph[index])

			gt_img = (gt_img > 0) * 1
			prob_img = (prob_img >= threshold) * 1

			intersection=intersection+np.sum(gt_img*prob_img)
			union=union+np.sum(gt_img)+np.sum(prob_img)-np.sum(gt_img*prob_img)
		iou=np.round(intersection/union,4)
		return iou

	def debug(self):
		"""
		show debug info
		:return: None
		"""
		print("Now enter debug mode....\nPlease check the info bellow:")
		print("total groundtruth: %d   total probgraph: %d\n"%(len(self.groundtruth),len(self.probgraph)))
		for index in range(len(self.groundtruth)):
			print(self.groundtruth[index],self.probgraph[index])
		print("Please confirm the groundtruth and probgraph name is opposite")


class DrawCurve:
	"""
	draw ROC/PR curve
	"""
	def __init__(self,savepath):
		self.savepath=savepath
		self.colorbar=['red','green','blue','black']
		self.linestyle=['-','-.','--',':','-*']

	def reload(self,xdata,ydata,auc,dataName,modelName):
		"""
		this function is to update data for Function roc/pr to draw
		:param xdata:  list,x-coord of roc(pr)
		:param ydata:  list,y-coord of roc(pr)
		:param auc:    numerical,area under curve
		:param dataName: string,name of dataset
		:param modelName: string,name of test model
		:return:  None
		"""
		self.xdata.append(xdata)
		self.ydata.append(ydata)
		self.modelName.append(modelName)
		self.auc.append(auc)
		self.dataName=dataName

	def newly(self,modelnum):
		"""
		renew all the data
		:param modelnum:  numerical,number of models to draw
		:return:  None
		"""
		self.modelnum = modelnum
		self.xdata = []
		self.ydata = []
		self.modelName = []
		self.auc = []

	def roc(self):
		"""
		draw ROC curve,save the curve graph to  savepath
		:return: None
		"""
		plt.figure(1)
		plt.title('ROC Curve of %s'%self.dataName, fontsize=15)
		plt.xlabel("False Positive Rate", fontsize=15)
		plt.ylabel("True Positive Rate", fontsize=15)
		plt.xlim(0, 1)
		plt.ylim(0, 1)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		for i in range(self.modelnum):
			plt.plot(self.xdata[i], self.ydata[i], color=self.colorbar[i%len(self.colorbar)], linewidth=2.0, linestyle=self.linestyle[i%len(self.linestyle)], label=self.modelName[i]+',AUC:' + str(self.auc[i]))
		plt.legend()
		plt.savefig(self.savepath+'%s_ROC.png'%self.dataName, dpi=800)


	def pr(self):
		"""
		draw PR curve,save the curve to  savepath
		:return: None
		"""
		plt.figure(2)
		plt.title('PR Curve of %s'%self.dataName, fontsize=15)
		plt.xlabel("Recall", fontsize=15)
		plt.ylabel("Precision", fontsize=15)
		plt.xlim(0, 1)
		plt.ylim(0, 1)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		for i in range(self.modelnum):
			plt.plot(self.xdata[i], self.ydata[i], color=self.colorbar[i%len(self.colorbar)], linewidth=2.0, linestyle=self.linestyle[i%len(self.linestyle)],label=self.modelName[i]+',MAP:' + str(self.auc[i]))
		plt.legend()
		plt.savefig(self.savepath+'%s_PR.png'%self.dataName, dpi=800)


def fileList(imgpath,filetype):
	return glob.glob(imgpath+filetype)


def drawCurve(gtlist,problist,modelName,dataset,savepath='./'):
	"""
	draw ROC PR curve,calculate AUC MAP IoU
	:param gtlist:  list,groundtruth list
	:param problist: list,list of probgraph list
	:param modelName:  list,name of test,model
	:param dataset: string,name of dataset
	:param savepath: string,path to save curve
	:return:
	"""
	assert(len(problist)==len(modelName))

	process = CollectData()
	painter_roc = DrawCurve(savepath)
	painter_pr = DrawCurve(savepath)
	modelNum=len(problist)
	painter_roc.newly(modelNum)
	painter_pr.newly(modelNum)

	# calculate param
	for index in range(modelNum):
		process.reload(gtlist,problist[index])
		# if filenames are not one-to-one match between gtlist and problist,the curve seems to be not right.
		(FPR, TPR, AUC), (Precision, Recall, MAP),IoU = process.statistics()
		painter_roc.reload(FPR, TPR, AUC,dataset, modelName[index])
		painter_pr.reload(Precision, Recall, MAP, dataset, modelName[index])

	# draw curve and save
	painter_roc.roc()
	painter_pr.pr()
