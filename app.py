from sklearn.datasets import fetch_covtype
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.title('Covtype dataset: exploring data')

###
@st.cache
def load_data():
    covtype_bunch = fetch_covtype()
    covtype_df = pd.DataFrame(data=np.c_[covtype_bunch['data'], covtype_bunch['target']],
                                columns=covtype_bunch['feature_names'] + ['covtype'])
    # since target classes are being parsed as float, we can convert them to categorical values
    covtype_df['covtype'] = covtype_df['covtype'].astype("int").astype("category")

    # convert the Soil_Type_* and Wilderness_Area_* columns in boolean types.
    covtype_df[[i for i in covtype_df.columns if i.startswith('Soil_Type')]] = \
    covtype_df[[i for i in covtype_df.columns if i.startswith('Soil_Type')]].astype(bool)

    covtype_df[[i for i in covtype_df.columns if i.startswith('Wilderness_Area_')]] = \
    covtype_df[[i for i in covtype_df.columns if i.startswith('Wilderness_Area_')]].astype(bool)
    return covtype_df

data_load_state = st.text('Loading data...')
covtype_df = load_data()
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(covtype_df)

col1, col2 = st.columns(2)

col1.subheader('Number of observations by covtype')
fig = plt.figure(figsize=(7, 6))
covtype_df.covtype.value_counts().plot(kind="bar", xlabel="covtype", ylabel="counts", rot=0)    
col1.pyplot(fig)

col2.subheader('Check covtype trends against dataset features')

# select which feature to plot
plot_feature = col2.selectbox(
    'Check the distribution of this feature according to covtype:',
    ('Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology'))

fig = plt.figure(figsize=(7, 5))
sns.violinplot(data=covtype_df, x=plot_feature, y="covtype")
col2.pyplot(fig)