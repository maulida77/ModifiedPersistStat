from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from scipy.stats import expon
import pickle
import numpy as np
import vectorization as vect
import pandas as pd
from auxiliary_functions import *
from numpy.random import choice, seed

number_labels = 50 
n_iters =100  
s=1
seed(s)

#%%
path_data = "Batik300/data/000/"
path_diag = "Batik300/pdiagrams/"
path_results = "results/"
data_name =  "batik300"

#%%
vec_parameters = dict()
vec_parameters['GetPersStats_modiv']=()
vec_parameters['GetPersStats']=()


#%%
complete = [
    {'base_estimator': ['RF'], 'n_estimators': [50,100,200,300,500]},
    {'base_estimator': ['SVM'], 'kernel': ['linear'], 'C': uniform(1,1000)},
    {'base_estimator': ['SVM'], 'kernel': ['rbf'], 'C': uniform(1,1000), 
      'gamma': expon(scale=.01)},
    {'base_estimator': ['SVM'], 'kernel': ['poly'], 'C': uniform(1,1000), 
      'degree': [2,3], 'gamma': expon(scale=.01)},
 ]

searchR =  RandomizedSearchCV(
    main_classifier(), param_distributions=complete, cv=3, n_iter=40, #modif here cv= 3 form 5
    return_train_score=True, scoring='accuracy', random_state=1
)

#%%
labels = range(50)

train_labels = np.array(pd.read_csv(path_data + "train.txt", sep=" ", 
                                    usecols=[0]).to_numpy().flatten().tolist()) #modif col 0 for batik, 1 for outex # double
test_labels = np.array(pd.read_csv(path_data + "test.txt", sep=" ", 
                                    usecols=[0]).to_numpy().flatten().tolist()) #modif col 0 for batik, 1 for outex #double
label_list = train_labels #-->modif for batik


train_index, test_index, y_train, y_test = train_test_split(range(len(label_list)), 
                                                    label_list, test_size=0.2, 
                                                    random_state=0)
index = train_index + test_index

#%%
pdiagrams = dict()
for i in index:
    pdiagrams["pdiag_l_d0_"+str(i)]= safe_load(path_diag + "l_d0_"+str(i))
    pdiagrams["pdiag_l_d1_"+str(i)]= safe_load(path_diag + "l_d1_"+str(i))
    pdiagrams["pdiag_u_d0_"+str(i)]= safe_load(path_diag + "u_d0_"+str(i))
    pdiagrams["pdiag_u_d1_"+str(i)]= safe_load(path_diag + "u_d1_"+str(i))
    

#%%
func_list = [getattr(vect, keys) for keys in vec_parameters.keys()]
print(func_list)
for func in func_list:
    feature_dictionary = dict()
    vec_methods = dict()
    vec_methods[func.__name__] = vec_parameters[func.__name__]
    func_parameters = load_parameters(func,vec_methods)
    
    feature_dictionary_l_d0 = feature_computation(vec_methods, pdiagrams, "pdiag_l_d0_",
                                                  train_index, test_index)
    feature_dictionary_l_d1 = feature_computation(vec_methods, pdiagrams, "pdiag_l_d1_",
                                                  train_index, test_index)
    feature_dictionary_u_d0 = feature_computation(vec_methods, pdiagrams, "pdiag_u_d0_",
                                                  train_index, test_index)
    feature_dictionary_u_d1 = feature_computation(vec_methods, pdiagrams, "pdiag_u_d1_",
                                                  train_index, test_index)
    
    for p in func_parameters:
        features= dict()
        for i in index:
            features[str(i)] = np.hstack(
                                    [
                                        feature_dictionary_l_d0[func.__name__+'_'+str(p)][str(i)],
                                        feature_dictionary_l_d1[func.__name__+'_'+str(p)][str(i)],
                                        feature_dictionary_u_d0[func.__name__+'_'+str(p)][str(i)],
                                        feature_dictionary_u_d1[func.__name__+'_'+str(p)][str(i)]
                                    ]
                                    )
        
        feature_dictionary[func.__name__+'_'+str(p)]= features

    with open(path_results+data_name+str(number_labels)+'_feature_'+func.__name__+'.pkl', 'wb') as f:
      pickle.dump(feature_dictionary, f)
      
      normalization = True 
      
    best_scores=parameter_optimization(train_index, y_train, vec_methods, feature_dictionary, 
                                      searchR, normalization)

    print("Parameter optimization:",best_scores)

    with open(path_results+data_name+str(number_labels)+'_best_scores_'+func.__name__+'.pkl', 'wb') as f:
      pickle.dump(best_scores, f)

    train_scores, test_scores = scores(train_index, y_train, test_index, y_test, 
                                       vec_methods, feature_dictionary, best_scores, 
                                       n_iters, normalization)

    print("The train accuracy is", train_scores)
    print("The test accuracy is", test_scores)
    
    with open(path_results+data_name+str(number_labels)+'_train_scores_'+func.__name__+'.pkl', 'wb') as f:
      pickle.dump(train_scores, f)
    
    with open(path_results+data_name+str(number_labels)+'_test_scores_'+func.__name__+'.pkl', 'wb') as f:
      pickle.dump(test_scores, f)
