#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Simplified Breast Cancer Selection</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Practice binary classification on Breast Cancer data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Exploratory Data Analysis</a></li>
#             <li> <a style="color:#303030" href="#P2">Modeling</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Binary Classification.
#     </div>
# </div>
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/17X_OTM8Zqg-r4XEakCxwU6VN1OsJpHh7?usp=sharing" title="momentum"> Assignment, Classification of breast cancer cells</a>
# </strong></nav>

# ## Submitted by Temiloluwa Ojo and Robiya Farmonova

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# In[ ]:


# get_ipython().system(u'sudo apt-get install build-essential swig')
# get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
# get_ipython().system(u'pip install -U auto-sklearn')
# get_ipython().system(u'pip install -U matplotlib')
# get_ipython().system(u'pip install pipelineprofiler')
# get_ipython().system(u'pip install shap')
# get_ipython().system(u'pip install --upgrade plotly')
# get_ipython().system(u'pip3 install -U scikit-learn')


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import plotly
plotly.__version__

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

# your code here
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay,mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression  
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import time
from google.colab import files

from sklearn import set_config
from sklearn.compose import ColumnTransformer

import autosklearn.classification
import PipelineProfiler
import shap
import datetime

from joblib import dump

import logging


# **Connect** to your Google Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[ ]:


data_path = "/content/drive/MyDrive/Introduction2DataScience/exercises/sit_w2d2_ml_engineering_assignment/data/raw/"


# In[ ]:


model_path = "/content/drive/MyDrive/Introduction2DataScience/exercises/sit_w2d2_ml_engineering_assignment/models/"


# In[ ]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[ ]:


logging.basicConfig(filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# Please Download the data from [this source](https://drive.google.com/file/d/1af2YyHIp__OdpuUeOZFwmwOvCsS0Arla/view?usp=sharing), and upload it on your introduction2DS/data google drive folder.

# <a id='P1' name="P1"></a>
# ## [Loading Data and Train-Test Split](#P0)
# 

# In[ ]:


df = pd.read_csv(f"{data_path}data-breast-cancer.csv")


# In[ ]:


#encode the categrical column
encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])


# In[ ]:


df.drop(['Unnamed: 32','id'], axis=1, inplace=True)


# In[ ]:


test_size = 0.2
random_state = 45


# In[ ]:


train, test = train_test_split(df, test_size=test_size, random_state=random_state)


# In[ ]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# In[ ]:


train.to_csv(f'{data_path}Breast_Cancer_Train.csv', index=False)


# In[ ]:


train= train.copy()


# In[ ]:


test.to_csv(f'{data_path}Breast_Cancer_Test.csv', index=False)


# In[ ]:


test = test.copy()


# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# In[ ]:


X_train, y_train = train.iloc[:,1:], train['diagnosis']


# In[ ]:


total_time = 600
per_run_time_limit = 30


# In[ ]:


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(X_train, y_train)


# In[ ]:


logging.info(f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[ ]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[ ]:


logging.info(f'Saved classification model at {model_path}model{timesstr}.pkl ')


# In[ ]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# In[ ]:


# profiler_data= PipelineProfiler.import_autosklearn(automl)
# PipelineProfiler.plot_pipeline_matrix(profiler_data)


# <a id='P2' name="P2"></a>
# ## [Model Evluation and Explainability](#P0)

# In[ ]:


X_test, y_test = train.iloc[:,1:], train['diagnosis'] 


# Now, we can attempt to predict the diagnosis prediction from our test set. To do that, we just use the .predict method on the object "automl" that we created and trained in the last sections:

# In[ ]:


y_pred = automl.predict(X_test)


# Let's now evaluate it using the mean_squared_error function from scikit learn:

# In[ ]:


logging.info(f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")


# we can also plot the y_test vs y_pred scatter:

# In[ ]:


df = pd.DataFrame(np.concatenate((X_test, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[ ]:


df.columns = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'Predicted Target','True Target']


# In[ ]:


fig = px.scatter(df, x='Predicted Target', y='True Target')
fig.write_html(f"{model_path}residualfig_{timesstr}.html")


# In[ ]:


logging.info(f"Figure of residuals saved as {model_path}residualfig_{timesstr}.html")


# #### Model Explainability

# In[ ]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test.iloc[:50, :], link = "identity")


# In[ ]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
# shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test.iloc[X_idx:X_idx+1,:], 
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[ ]:


shap_values = explainer.shap_values(X = X_test.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
# shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = X_test.iloc[0:50,:],
                  show=False)
plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
logging.info(f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")


# --------------
# # End of This Notebook
