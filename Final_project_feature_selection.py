
# coding: utf-8

# In[762]:


# copyright: liu792@usc.edu

import pandas as pd
import sys
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import logger
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# In[699]:



def lassoSelection(X_train, y_train, n):
	'''
	Lasso feature selection.  Select n features. 
	'''
	#lasso feature selection
	#print (X_train)
	clf = LassoCV()
	sfm = SelectFromModel(clf, threshold=0)
	sfm.fit(X_train, y_train)
	X_transform = sfm.transform(X_train)
	n_features = X_transform.shape[1]
	
	#print(n_features)
	while n_features > n:
		sfm.threshold += 0.01
		X_transform = sfm.transform(X_train)
		n_features = X_transform.shape[1]
	features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
	logger.info("selected features are {}".format(features))
	return features


# In[606]:


def REF_LR(array, k):
    X = array[:,1:1085]
    X = X.astype('double')
    Y = array[:,1085]
    Y = Y.astype('int')
    # feature extraction
    model = LogisticRegression()
    rfe = RFE(model, k)
    fit = rfe.fit(X, Y)
    selected_index = []
    for idx, val in enumerate(fit.support_):
        if val:
            print(idx)
            selected_index.append(idx)
    selected_features = []
    for i in selected_index:
        selected_features.append(feature_list[i])
    return selected_features


# In[607]:


def REF_RF(array, k):
    X = array[:,1:1085]
    X = X.astype('double')
    Y = array[:,1085]
    Y = Y.astype('int')
    # feature extraction
    model = RandomForestClassifier(n_estimators=100)
    rfe = RFE(model, k)
    fit = rfe.fit(X, Y)
    selected_index = []
    for idx, val in enumerate(fit.support_):
        if val:
            print(idx)
            selected_index.append(idx)
    selected_features = []
    for i in selected_index:
        selected_features.append(feature_list[i])
    return selected_features


# In[608]:


def univariate_selection(array, num_para, method):
    X = array[:,1:1085]
    X = X.astype('double')
    Y = array[:,1085]
    Y = Y.astype('int')
    feature_list = list(df_pro_gender)
    feature_list = feature_list[1:]
    # feature extraction
    test = SelectKBest(score_func=method, k=num_para)
    fit = test.fit(X, Y)
    # summarize scores
    np.set_printoptions(precision=3)
    #print(fit.scores_)
    features = fit.transform(X)
    # summarize selected features
    #print(features[0:5,:])
    #print(fit.scores_)
    selected_index = fit.scores_.argsort()[-num_para:][::-1]
    selected_features = []
    for i in selected_index:
        selected_features.append(feature_list[i])
    return selected_features


# In[609]:


def kmeans_cluster_scan(df):
#kmeans clustering to see if gender/msi can be distinguished from each of the proteomic data
    num_cols = len(df.columns)
    for x in xrange(1, num_cols):
        sys.stdout.write('\r'+str(x))
        sys.stdout.flush()
        y = x + 1
        cluster = KMeans(n_clusters = 2)
        df["cluster"] = cluster.fit_predict(df[df.columns[x:y]])
        df_count = df.groupby([0, "cluster"]).size().reset_index(name="time")
        if (df_count.time > 1).any():
            #print "scan stopped"
            print("th column")
            print(df_count)
    print ("")
    


# In[610]:


def model_fit_predict(X_train,X_test,y_train,y_test):

	np.random.seed(2018)
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.svm import SVC
	from sklearn.metrics import precision_score
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import recall_score
	models = {
		'LogisticRegression': LogisticRegression(),
		'ExtraTreesClassifier': ExtraTreesClassifier(),
		'RandomForestClassifier': RandomForestClassifier(),
    	'AdaBoostClassifier': AdaBoostClassifier(),
    	'GradientBoostingClassifier': GradientBoostingClassifier(),
    	'SVC': SVC()
	}
	tuned_parameters = {
		'LogisticRegression':{'C': [1, 10]},
		'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
		'RandomForestClassifier': { 'n_estimators': [16, 32] },
    	'AdaBoostClassifier': { 'n_estimators': [16, 32] },
    	'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    	'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
	}
	scores= {}
	for key in models:
		clf = GridSearchCV(models[key], tuned_parameters[key], scoring=None,  refit=True, cv=10)
		clf.fit(X_train,y_train)
		y_test_predict = clf.predict(X_test)
		precision = precision_score(y_test, y_test_predict)
		accuracy = accuracy_score(y_test, y_test_predict)
		f1 = f1_score(y_test, y_test_predict)
		recall = recall_score(y_test, y_test_predict)
		specificity = specificity_score(y_test, y_test_predict)
		scores[key] = [precision,accuracy,f1,recall,specificity]
	#print(scores)
	return scores


# In[611]:


def specificity_score(y_true, y_predict):
	'''
	true_negative rate
	'''
	true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
	real_negative = len(y_true) - sum(y_true)
	return true_negative / real_negative 


# In[612]:


def draw(scores):
	'''
	draw scores.
	'''
	logger.info("scores are {}".format(scores))
	ax = plt.subplot(111)
	precisions = []
	accuracies =[]
	f1_scores = []
	recalls = []
	categories = []
	specificities = []
	N = len(scores)
	ind = np.arange(N)  # set the x locations for the groups
	width = 0.1        # the width of the bars
	for key in scores:
		categories.append(key)
		precisions.append(scores[key][0])
		accuracies.append(scores[key][1])
		f1_scores.append(scores[key][2])
		recalls.append(scores[key][3])
		specificities.append(scores[key][4])

	precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
	accuracy_bar = ax.bar(ind+1*width, accuracies,width=0.1,color='g',align='center')
	f1_bar = ax.bar(ind+2*width, f1_scores,width=0.1,color='r',align='center')
	recall_bar = ax.bar(ind+3*width, recalls,width=0.1,color='y',align='center')
	specificity_bar = ax.bar(ind+4*width,specificities,width=0.1,color='purple',align='center')

	print(categories)
	ax.set_xticks(np.arange(N))
	ax.set_xticklabels(categories)
	ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0],specificity_bar[0]), ('precision', 'accuracy','f1','sensitivity','specificity'))
	ax.grid()
	plt.show()


# In[613]:


def find_popular(list, num_selected):
    best_overlapping_features = []
    feature_frequency_dict = {}
    for f in list:
        if f in feature_frequency_dict:
            feature_frequency_dict[f] += 1
        else:
            feature_frequency_dict[f] = 1
    s = [(k, feature_frequency_dict[k]) for k in sorted(feature_frequency_dict, key=feature_frequency_dict.get, reverse=True)]
    for i in range(num_selected):
        best_overlapping_features.append(s[i][0])
    return best_overlapping_features


# In[614]:


def knn(X_train, X_test, y_train, y_test, k):
    ## Import the Classifier.
    from sklearn.neighbors import KNeighborsClassifier
    ## Instantiate the model with 5 neighbors. 
    knn = KNeighborsClassifier(n_neighbors=k)
    ## Fit the model on the training data.
    knn.fit(X_train, y_train)
    ## See how the model performs on the test data.
    print(knn.score(X_test, y_test))
    


# In[776]:


def rank_sort(feature_lists, num_selected):
    rank_dict = {}
    for list in feature_lists:
        for index,feature in enumerate(list):
            if feature not in rank_dict:
                rank_dict[feature] = index
            else:
                rank_dict[feature] += index
    s = [(k, rank_dict[k]) for k in sorted(rank_dict, key=rank_dict.get)]
    print(s)
    rank_sorted_list = []
    for i in range(num_selected):
        rank_sorted_list.append(s[i][0])
    return rank_sorted_list
        
            


# In[760]:


def ElasticNet_selector(X_train, X_test, y_train, y_test, alpha):
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import r2_score
    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
    y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
    r2_score_enet = r2_score(y_test, y_pred_enet)
    print(enet)
    print("r^2 on test data : %f" % r2_score_enet)
    plt.plot(enet.coef_, color='lightgreen', linewidth=2,
    label='Elastic net coefficients')
    plt.plot(lasso.coef_, color='gold', linewidth=2,
             label='Lasso coefficients')
    plt.plot(coef, '--', color='navy', label='original coefficients')
    plt.legend(loc='best')
    plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
    plt.show()


# In[834]:


def merge_list(lists):
    merged_list = []
    for i in range(len(lists[0])):
        for l in lists:
            if l[i] not in merged_list:
                merged_list.append(l[i])
    print(merged_list)
    return merged_list


# In[639]:


df_pro = pd.read_csv("/Users/RyanLiu/Desktop/EE542_final_project/train_pro.tsv", sep="\t")
df_pro['count_nan'] = df_pro.isnull().sum(axis=1)
df_pro_perfect = df_pro[df_pro.count_nan == 0]
df_pro_perfect.pop('count_nan')
df_pro_perfect = df_pro_perfect.transpose()
df_pro_perfect.insert(0, 'sample', df_pro_perfect.index)
df_cli = pd.read_csv('/Users/RyanLiu/Desktop/EE542_final_project/train_cli_corrected.tsv', sep="\t")
df_label = pd.read_csv('/Users/RyanLiu/Desktop/EE542_final_project/sum_tab_1.csv')
df_cli_label = pd.merge(df_cli, df_label, on="sample")
df_cli_label = df_cli_label.drop(['mismatch'], axis = 1)
df_final = pd.merge(df_pro_perfect, df_cli_label, on='sample')
df_pro_gender = df_final.drop(['msi'], axis = 1)
df_pro_msi = df_final.drop(['gender'], axis = 1)
# 0-1 of pro_gender
df_pro_gender['gender'] = df_pro_gender['gender'].map({'Male': 1, 'Female': 0})
df_pro_msi['msi'] = df_pro_msi['msi'].map({'MSI-High' : 1, 'MSI-Low/MSS' : 0})


# In[867]:


feature_list = list(df_pro_gender)
feature_list = feature_list[1:]
gender_best_feature_us_fclass = univariate_selection(df_pro_gender.values, 100, f_classif)
gender_best_feature_us_chi2 = univariate_selection(df_pro_gender.values, 100, chi2)
gender_best_feature_us_f_regression = univariate_selection(df_pro_gender.values, 100, f_regression)
gender_best_feature_us_mutual_info_regression = univariate_selection(df_pro_gender.values, 100, mutual_info_regression)
gender_best_feature_rfe_LR = REF_LR(df_pro_gender.values, 100)
gender_best_feature_rfe_RF = REF_RF(df_pro_gender.values, 100)
#msi_best_feature_us_fclass = univariate_selection(df_pro_msi.values, 10, f_classif)
#msi_best_feature_us_chi2 = univariate_selection(df_pro_msi.values, 10, chi2)
#msi_best_feature_rfe_LR = REF_LR(df_pro_msi.values, 10)
#msi_best_feature_rfe_RF = REF_RF(df_pro_msi.values, 10)
print(gender_best_feature_us_fclass)
print(gender_best_feature_us_chi2)
print(gender_best_feature_us_f_regression)
print(gender_best_feature_us_mutual_info_regression)
print(gender_best_feature_rfe_LR)
print(gender_best_feature_rfe_RF)


# In[894]:


#pre-processing: 1085 -> 200
myList = [gender_best_feature_us_fclass, gender_best_feature_us_chi2, gender_best_feature_us_f_regression, gender_best_feature_us_mutual_info_regression, gender_best_feature_rfe_LR, gender_best_feature_rfe_RF]
#rank_sorted_list = rank_sort(myList, 310)
merged_list = merge_list(myList)


# In[911]:


len(merged_list)


# In[938]:


test_list = []

test_list = gender_best_feature_us_fclass + ['gender']
#print(test_list)
test_df = df_pro_gender[test_list]
#print(test_df)
y_data = test_df.pop('gender').values
columns = test_df.columns
X_data = test_df.values
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
#standardize the data.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
n = 22
feaures_columns = lassoSelection(X_train, y_train, n)
selected_feature = []
for i in feaures_columns:
    selected_feature.append(test_list[i])
print(selected_feature)
scores = model_fit_predict(X_train[:,feaures_columns],X_test[:,feaures_columns],y_train,y_test)
knn(X_train,X_test,y_train,y_test,5)
get_ipython().run_line_magic('matplotlib', 'inline')
draw(scores)


# In[50]:


#n = 5
#feaures_columns = lassoSelection(X_train, y_train, n)


# In[335]:





# In[198]:


draw(scores)


# In[156]:



draw(scores)


# In[159]:


draw(scores)


# In[170]:


draw(scores)


# In[174]:


draw(scores)

