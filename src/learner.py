from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import Boost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import  f1_score, roc_auc_score

from data import *
from visualize import *
from logs import log


logger = log("../logs/log")

class_name = ["Health", "Cholera"]


def fit_model(df, model, parameters, cv=5, model_name="RF"):

    y=df['LabResult']
    X = df.drop(['LabResult'],axis=1) 

    logger.info("fitting model")
    clf = GridSearchCV(model, parameters, cv=cv, scoring='f1_micro')
    clf.fit(X, y)
    be = clf.best_estimator_
    y_pred = be.predict(X)
    cm = confusion_matrix(y.values, y_pred)
    print(cm)
    f1_tra=f1_score(y.values, y_pred)
    roc=roc_auc_score(y.values, y_pred)
    logger.info(f"Train_fscore:{f1_tra}: Train_roc:{roc}")
    joblib.dump(clf.best_estimator_, '../models/{}.pkl'.format(model_name))

    #return y_pred, y_test
def main():
    df=pd.read_csv('../data/train_data.csv')
    categ_column=["District", "WasteWater", "Year"]
    numeric_column=["Humidity", "Wind_Dir", 'Temp_mean', 'Rainfall']
    df_pipeline=data_pipeline(numeric_column, categ_column)
    clf = Pipeline(steps=[('preprocessor', df_pipeline),
                      ('classifier', RandomForestClassifier())])
    parameters=[{'classifier__n_estimators':[4,8,16,32,100,250], 
           'classifier__max_features': ['auto','log2',None], 
           'classifier__min_samples_leaf': [0.2,0.4,1], 
           'classifier__max_depth': [27]
           
           
}]
    fit_model(df, clf, parameters, cv=5)  

    
        


if __name__ == "__main__":
    main()
            

    

