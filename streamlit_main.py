import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pickle

# Function for loading pickle with model and pickle with encoder
@st.cache_data
def load_pickles():
    model = pickle.load(open("cc_model.pkl", "rb"))
    encoder_dict = pickle.load(open("cc_label_encoders.pkl", "rb"))
    return model, encoder_dict



st.title('Title')

st.header('Problem Description')
st.write('Description')

st.header('Prediction')
st.write('Description')

# Load data
cc_data = pd.read_csv("data/cc_approvals.data", header=None)
column_names = ["Gender",
                "Age",
                "Debt",
                "Married",
                "BankCustomer",
                "EducationLevel",
                "Ethnicity",
                "YearsEmployed",
                "PriorDefault",
                "Employed",
                "CreditScore",
                "DriversLicense",
                "Citizen",
                "ZipCode",
                "Income",
                "ApprovalStatus"]
# Add column names
cc_data.columns = column_names
#define mask to filter only for values we are interested in
mask = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore',
        'DriversLicense', 'Citizen', 'Income'] 
# apply mask
X = cc_data[mask]
y = cc_data["ApprovalStatus"]
# Replace '?' with NaN and drop NaN
X = X.replace('?', np.NaN)
X = X.dropna(axis=0)
y = y[X.index]
# converting "Total Charges" to numeric
X["Age"] = X["Age"].astype(float)

data = X.iloc[0, :].to_dict()
# The dtypes
dtypes = X.dtypes.to_dict()
# max and min values for the numerical columns
max_values = X.max().to_dict()
min_values = X.min().to_dict()

# Creating the input fields using the data of the first row of the train data as default values
input_fields = {}
for col in X.columns:
    if dtypes[col] == "int64":
        input_fields[col] = st.slider(
            col, min_value=min_values[col], max_value=max_values[col], value=data[col]
        )
    elif dtypes[col] == "float64":
        input_fields[col] = st.slider(
            col, min_value=min_values[col], max_value=max_values[col], value=data[col]
        )
    else:
        input_fields[col] = st.selectbox(col, X[col].unique())

df_input = pd.DataFrame([input_fields])
st.write(df_input)

# Load model and encoder
model, encoder_dict = load_pickles()

# Apply encoder to input data
for col in df_input.columns:
    if col in list(encoder_dict.keys()):
        column_le = encoder_dict[col]
        df_input.loc[:, col] = column_le.transform(df_input.loc[:, col])
    else:
        continue

# Make prediction
prediction = model.predict(df_input)

st.write(prediction)