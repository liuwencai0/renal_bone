import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler

#应用标题
st.title('A machine learning-based predictive model for predicting bone metastasis in patients with renal cancer')

# conf
st.sidebar.markdown('## Variables')
#Age = st.sidebar.slider("Age", 1, 99, value=25, step=1)
#Laterality = st.sidebar.selectbox('Laterality',('Left','Right','Other'),index=0)
Race = st.sidebar.selectbox("Race",('White','Black','Other'))
Sex = st.sidebar.selectbox("Sex",('Male','Female'))
Laterality = st.sidebar.selectbox("Laterality",('Left','Right','Not a paired site'))
Grade = st.sidebar.selectbox("Grade",('Well differentiated','Moderately differentiated','Poorly differentiated'
                                      ,'Undifferentiated; anaplastic','Unknow'))
T = st.sidebar.selectbox("T stage",('T1','T2','T3','T4','TX'))
N = st.sidebar.selectbox("N stage",('N0','N1','NX'))
Liver_metastasis = st.sidebar.selectbox("Liver.metastasis",('No','Yes'),index=0)
Lung_metastasis = st.sidebar.selectbox("Lung.metastasis",('No','Yes'),index=0)
#Chemotherapy = st.sidebar.selectbox("Chemotherapy",('No','Yes'),index=0)
Sequence_number = st.sidebar.selectbox("Sequence.number",('One primary only','1st primaries','2nd primaries','more'))
#Lung_metastases = st.sidebar.selectbox("Lung metastases",('No','Yes'))

# str_to_int

map = {'Left':0,'Right':1,'Not a paired site':2,'White':0,'Black':1,'Other':2,
       'T1':0,'T2':1,'T3':2,'T4':3,'TX':4,'N0':0,'N1':1,'NX':2,'No':0,'Yes':1,'Male':0,'Female':1,
       'One primary only':0,'1st primaries':1,'2nd primaries':2,'more':3,
       'Well differentiated':0,'Moderately differentiated':1,'Poorly differentiated':2,'Undifferentiated; anaplastic':3,
       'Unknow':4}
#Age =map[Age]
Laterality =map[Laterality]
Sex =map[Sex]
Race =map[Race]
Grade =map[Grade]
T =map[T]
N =map[N]
#surgery =map[surgery]
#Radiation =map[Radiation]
Sequence_number =map[Sequence_number]
Liver_metastasis =map[Liver_metastasis]
Lung_metastasis =map[Lung_metastasis]

# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
thyroid_train['Bone.metastases'] = thyroid_train['Bone.metastases'].apply(lambda x : +1 if x==1 else 0)
#thyroid_test = pd.read_csv('test.csv', low_memory=False)
#thyroid_test['BM'] = thyroid_test['BM'].apply(lambda x : +1 if x==1 else 0)
features = ['Race','Sex','Laterality','Grade','T','N','Liver.metastasis','Lung.metastasis','Sequence.number']
target = 'Bone.metastases'

#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

XGB = XGBClassifier(random_state=32,max_depth=5,n_estimators=31)
XGB.fit(X_ros, y_ros)


sp = 0.5
#figure
is_t = (XGB.predict_proba(np.array([[Race,Sex,Laterality,Grade,T,N,Liver_metastasis,Lung_metastasis,Sequence_number]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[Race,Sex,Laterality,Grade,T,N,Liver_metastasis,Lung_metastasis,Sequence_number]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for BM:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of BM:  '+str(prob)+'%')
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行



st.title("")
st.title("")
st.title("")
st.title("")
#st.warning('This is a warning')
#st.error('This is an error')

#st.info('Information of the model: Auc: 0.737 ;Accuracy: 0.784 ;Sensitivity(recall): 0.550 ;Specificity :0.839 ')
#st.success('Affiliation: The First Affiliated Hospital of Nanchang University, Nanchnag university. ')





