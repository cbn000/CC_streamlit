import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Function for loading pickle with model and pickle with encoder
# @st.cache_data


def load_pickles():
    model = pickle.load(open("cc_model.pkl", "rb"))
    encoder_dict = pickle.load(open("cc_label_encoders.pkl", "rb"))
    return model, encoder_dict

# Function for loading the data


@st.cache_data
def load_data(data_file_path):
    cc_data = pd.read_csv(data_file_path, header=None)
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
    return cc_data

# function for preprocessing the data


def preprocess_data(dataframe):
    mask = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'YearsEmployed', 'PriorDefault',
            'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'Income']
    # apply mask
    X = dataframe[mask]
    y = dataframe["ApprovalStatus"]
    # Replace '?' with NaN and drop NaN
    X = X.replace('?', np.NaN)
    X = X.dropna(axis=0)
    y = y[X.index]
    # converting "Total Charges" to numeric
    X["Age"] = X["Age"].astype(float)
    return X, y

# function for applying the encoder


def apply_encoder(dataframe, encoder_dict):
    for col in dataframe.columns:
        if col in list(encoder_dict.keys()):
            column_encoder = encoder_dict[col]
            dataframe.loc[:, col] = column_encoder.transform(
                dataframe.loc[:, col])
        else:
            continue
    return dataframe


# Load data
cc_data = load_data("data/cc_approvals.data")
# Preprocess data
X, y = preprocess_data(cc_data)
# Load model and encoder
model, encoder_dict = load_pickles()


# Body of the app
st.title('Credit Card Approval')

st.header('Credit Card Approval Form')

st.write('Please enter customer data:')

# Creating the input fields using the data of the first row of the train data as default values
data_row0 = X.iloc[0, :].to_dict()
# The dtypes
dtypes = X.dtypes.to_dict()
# max and min values for the numerical columns
max_values = X.max().to_dict()
min_values = X.min().to_dict()
# Generate input fields in loop
input_fields = {}
for col in X.columns:
    if dtypes[col] == "int64":
        input_fields[col] = st.slider(
            f"Please enter {col} value for customer:",
            min_value=min_values[col], max_value=max_values[col], value=data_row0[col],
        )
    elif dtypes[col] == "float64":
        input_fields[col] = st.slider(
            f"Please enter {col} value for customer:",
            min_value=min_values[col], max_value=max_values[col], value=data_row0[col]
        )
    else:
        input_fields[col] = st.selectbox(f"Please enter {col} value for customer:",
                                         X[col].unique())

# Inputs to dataframe
df_input = pd.DataFrame([input_fields])
# st.write(df_input)

# Apply encoder to input data
df_input = apply_encoder(df_input, encoder_dict)

# Make prediction
prediction = model.predict(df_input)

# st.write(prediction)
# Building a indicator for the prediction
if prediction == 1:
    st.success('Approve')
else:
    st.error('Reject!')


st.header('Data Description')
# Approval ratio per Gender
approval_ethnicity = cc_data.groupby('Gender')['ApprovalStatus'].value_counts(
    normalize=True).to_frame().reset_index()
data_plot = approval_ethnicity[approval_ethnicity['ApprovalStatus']
                               == '+'].drop('ApprovalStatus', axis=1)
fig = px.bar(data_plot, x='Gender', y='proportion',
             title='Approval Ratio per Gender')
st.plotly_chart(fig)

# Approval ratio per Ethnicity
approval_ethnicity = cc_data.groupby('Ethnicity')['ApprovalStatus'].value_counts(
    normalize=True).to_frame().reset_index()
data_plot = approval_ethnicity[approval_ethnicity['ApprovalStatus']
                               == '+'].drop('ApprovalStatus', axis=1)
fig = px.bar(data_plot, x='Ethnicity', y='proportion',
             title='Approval Ratio per Ethnicity')
st.plotly_chart(fig)

# Median Income per Ethnicity
# Mean of income per ethnicity
data_plot = cc_data.groupby('Ethnicity')['Income'].mean()
print(data_plot.sort_values(
    ascending=False).to_frame().reset_index().loc[1, "Income"])
# plot with maximum of 1234 on y axis
fig = px.bar(data_plot,
             range_y=(0, data_plot.sort_values(
                 ascending=False).to_frame().reset_index().loc[1, "Income"]*1.1)
             )
st.plotly_chart(fig)
