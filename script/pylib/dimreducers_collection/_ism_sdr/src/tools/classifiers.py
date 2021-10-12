
from sklearn import svm
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score


def use_svm(X,Y,k='rbf', K=None):	
	svm_object = svm.SVC(kernel=k)

	if K is None:
		svm_object.fit(X, Y)
		out_allocation = svm_object.predict(X)
	else:
		svm_object.fit(K, Y)
		out_allocation = svm_object.predict(K)

	#nmi = normalized_mutual_info_score(out_allocation, Y)
	acc = accuracy_score(out_allocation, Y)

	return [out_allocation, acc, svm_object]


def apply_svm(X,Y, svm_obj):
	out_allocation = svm_obj.predict(X)
	#nmi = normalized_mutual_info_score(out_allocation, Y)
	acc = accuracy_score(out_allocation, Y)
	return acc

