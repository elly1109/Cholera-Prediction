import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import train_test_split
np.random.seed(0)

feature_columns = ['District','Year','Rainfall', 'Temp_max', 'Temp_min', 'Temp_mean',
       'Temp_range', 'Humidity', 'Wind_Dir', 'WasteWater']

def load_data(file_name):
       """
       data loading
       """
       df = pd.read_csv(file_name)
       return df

def load_test_data(file_name, label_name):
       feature = pd.read_csv(file_name)
       label   = pd.read_csv(label_name)

       return feature, label

def get_balanced_data(X, y):
       smote = SMOTEENN(random_state=0)
       X, y = smote.fit_sample(X, y)    
       return X, y     


def data_pipeline(numeric_feature, categorical_feature):
       numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())])


       categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

       preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_feature),
                     ('cat', categorical_transformer, categorical_feature)])
       return preprocessor


if __name__ == "__main__":
    df=pd.read_csv('../data/train_data.csv')
    categ_column=["District", "WasteWater", "Year"]
    numeric_column=["Humidity", "Wind_Dir", 'Temp_mean', 'Rainfall']
    df_pipeline=data_pipeline(numeric_column, categ_column)
    print(df.head())


