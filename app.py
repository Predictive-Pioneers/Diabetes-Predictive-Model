import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# load the diabetes dataset
diabetes_df = pd.read_csv(r'D:\Internship(yuvaintern)\week1\Project\Database\diabetes.csv')

# group the data by outcome to get a sense of the distribution
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# split the data into input and target variables
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# scale the input variables using StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create an SVM model with a linear kernel
model = svm.SVC(kernel='linear')

# train the model on the training set
model.fit(X_train, y_train)

# make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)

# create the Streamlit app
def app():

    img = Image.open(r"img.jpeg")
    img = img.resize((200,200))
    st.image(img,width=200)


    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Enter Patient Details')

    preg = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=6, step=1)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=199, value=148)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=72)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=99, value=35)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=0)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=33.6, format="%.1f")
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.627, format="%.3f")
    age = st.sidebar.number_input('Age', min_value=21, max_value=81, value=50, step=1)


    # make a prediction based on the user input
    if st.sidebar.button('Predict'):
        input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
        input_data_nparray = np.asarray(input_data)
        reshaped_input_data = input_data_nparray.reshape(1, -1)

        scaled_input = scaler.transform(reshaped_input_data)
        prediction = model.predict(scaled_input)

        if prediction == 1:
            st.warning('This person has diabetes.')
        else:
            st.success('This person does not have diabetes.')

    # display some summary statistics about the dataset
    st.header('Dataset Summary')
    st.write(diabetes_df.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)

    # display the model accuracy
    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')

if __name__ == '__main__':
    app()