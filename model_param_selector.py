def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))


def model_optimizer(model_name , X_train_scaled , y_train):
  '''
  Finds best model parameters for the required model using X_trained_scaled 
  and y_train
  
  '''
  from sklearn.model_selection import cross_val_score
  from sklearn.naive_bayes import GaussianNB
  from sklearn.linear_model import LogisticRegression
  from sklearn import tree
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  from sklearn.model_selection import GridSearchCV 
  from sklearn.model_selection import RandomizedSearchCV 
  from xgboost import XGBClassifier
  

  if model_name ==' KNeighborsClassifier':
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors' : [3,5,7,9],
                  'weights' : ['uniform', 'distance'],
                  'algorithm' : ['auto', 'ball_tree','kd_tree'],
                  'p' : [1,2]}
    clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
    best_clf_knn = clf_knn.fit(X_train_scaled,y_train)
    return clf_performance(best_clf_knn,'KNN')
  
  elif model_name == 'logisticRegression':
    lr = LogisticRegression()
    param_grid = {'max_iter' : [2000],
                  'penalty' : ['l1', 'l2'],
                  'C' : np.logspace(-4, 4, 20),
                  'solver' : ['liblinear']}

    clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
    best_clf_lr = clf_lr.fit(X_train_scaled,y_train)
    return clf_performance(best_clf_lr,'Logistic Regression')

  elif model_name == 'SVC':
    svc = SVC(probability = True)
    param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                      'C': [.1, 1, 10, 100, 1000]},
                                    {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                    {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
    clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
    best_clf_svc = clf_svc.fit(X_train_scaled,y_train)
    return clf_performance(best_clf_svc,'SVC')

  elif model_name=='RandomForestClassifier':
    rf = RandomForestClassifier(random_state = 1)
    param_grid =  {'n_estimators': [400,450,500,550],
                  'criterion':['gini','entropy'],
                                      'bootstrap': [True],
                                      'max_depth': [15, 20, 25],
                                      'max_features': ['auto','sqrt', 10],
                                      'min_samples_leaf': [2,3],
                                      'min_samples_split': [2,3]}
                                      
    clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
    best_clf_rf = clf_rf.fit(X_train_scaled,y_train)
    
    return clf_performance(best_clf_rf,'Random Forest')

  elif model_name =='XGBClassifier':
    xgb = XGBClassifier(random_state = 1)

    param_grid = {
        'n_estimators': [450,500,550],
        'colsample_bytree': [0.75,0.8,0.85],
        'max_depth': [None],
        'reg_alpha': [1],
        'reg_lambda': [2, 5, 10],
        'subsample': [0.55, 0.6, .65],
        'learning_rate':[0.5],
        'gamma':[.5,1,2],
        'min_child_weight':[0.01],
        'sampling_method': ['uniform']
    }

    clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
    best_clf_xgb = clf_xgb.fit(X_train_scaled,y_train)
    return clf_performance(best_clf_xgb,'XGB')
  else:
    return 0
