from  .preprocess import *

import pandas as pd
from sklearn.model_selection import train_test_split

path = 'C:/QamerAI/100 days of ML by Campus x/Portfolio Projects/student performance/data/myData.csv'
df = pd.read_csv(path)



def model_train(model):
    from sklearn.metrics import r2_score
    preprocess_ = preprocessed(df)
    X = preprocess_.drop('Performance Index', axis=1)
    y = preprocess_['Performance Index']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    score = r2_score(y_test,model_pred)

    return score
