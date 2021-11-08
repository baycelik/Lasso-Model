import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

data=pd.read_csv(r"../input/consumption-sao-paulo/Consumo_cerveja.csv")

data.head()

data.columns

data.dtypes

data.replace(",",".",inplace=True,regex=True)

data.drop("Data",1,inplace=True)

data=data.dropna()

data=data.astype("float64")

data.dtypes

y=data["Consumo de cerveja (litros)"]

X=data.drop("Consumo de cerveja (litros)",1)

X.columns

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=42)

lasso_model=Lasso(alpha=0.1).fit(X_train,y_train)

lasso_model

lasso_model.coef_

lasso=Lasso()

lambdas=10**np.linspace(10,-2,100)*0.5

mycoefs=[]


for i in lambdas:
    lasso.set_params(alpha=i)
    lasso.fit(X_train,y_train)
    mycoefs.append(lasso.coef_)
    
    
ax=plt.gca()
ax.plot(lambdas,mycoefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

lasso_model.predict(X_test)

y_pred=lasso_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

lasso_cv_model=LassoCV(alphas=None,cv=10,max_iter=10000,normalize=True)

lasso_cv_model.fit(X_train,y_train)

lasso_cv_model.alpha_

lasso_tuned=Lasso(alpha=lasso_cv_model.alpha_)

lasso_tuned.fit(X_train,y_train)

y_pred=lasso_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

