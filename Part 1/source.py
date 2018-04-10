from FDA import fda
from Logistic import logistic
import matplotlib.pyplot as plt

if __name__ == "__main__":
	print("Hello")
	fda_model = fda('data.txt')
	logi_model = logistic('data.txt')


	fda_model.build()
	print(fda_model.evaluate())

	logi_model.build()
	print(logi_model.evaluate())




	fda_fpr, fda_tpr, fda_th, fda_roc_auc = fda_model.ROC()
	logi_fpr, logi_tpr, logi_th, logi_roc_auc = logi_model.ROC()



	plt.title('Receiver Operating Characteristic')
	plt.plot(fda_fpr, fda_tpr, 'b', label = 'FDA AUC = %0.2f' % fda_roc_auc)
	plt.plot(logi_fpr, logi_tpr, 'g', label = 'Logistic AUC = %0.2f' % logi_roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1.05])
	plt.ylim([0, 1.05])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()