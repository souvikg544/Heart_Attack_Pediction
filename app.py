import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def hometab():
    st.title("Heart Attack Predictor")
    st.subheader("Sit with your reports and fill the following..")
    st.image("data//heart.jpg",width=500)


    age=st.number_input("Age")
    sex=st.selectbox("Gender",["Male","Female"])
    pain=st.selectbox("Chest Pain Type",["typical angina","atypical angina","non-anginal pain","asymptomatic"])
    pressure=st.number_input("Resting Blood Pressure")
    chol=st.number_input("Cholestoral in mg/dl fetched via BMI sensor")
    sugar=st.selectbox("Do you have Sugar(blood sugar > 120 mg/dl)",["Yes","No"])
    ecg=st.selectbox("Resting electrocardiographic results",["Normal","having ST-T wave abnormality"," left ventricular hypertrophy by Estes' criteria"])
    Rate=st.number_input("Maximum heart rate achieved")
    exercise=st.selectbox("Exercise induced problem ?",["Yes","No"])
    oldpeak=st.number_input("Old peak(Old peak = ST depression induced by exercise relative to rest	)")
    slope=st.number_input("Slope(0-2)(the ST/heart rate)")
    vessel=st.number_input("Major Blood Vessels")
    thal=st.number_input("Thal Rate")

    if(sex == "Male"):
        sex=1
    else:
        sex=0

    if(pain == "typical angina"):
        pain=1
    elif(pain== "atypical angina"):
        pain=2
    elif(pain== "non-anginal pain"):
        pain=3
    elif(pain=="asymptomatic"):
        pain=4

    if(sugar == "Yes"):
        sugar=1
    else:
        sugar=0

    if(ecg == "normal"):
        ecg=0
    elif(ecg=="having ST-T wave abnormality"):
        ecg=1
    else:
        ecg=2

    if(exercise == "Yes"):
        exercise=1
    else:
        exercise=0

    X=df.drop(['output'],axis=1)
    y=df['output']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.12, random_state=0)
    log=LogisticRegression()
    log.fit(X_train,y_train)
    arr=log.predict([[age,sex,pain,pressure,chol,sugar,ecg,Rate,exercise,oldpeak,slope,vessel,thal]])

    if(st.button("Predict")):
        if(arr[0] == 0):
            st.success("There is less chance of Heart attack..Do visit a doctor if you feel any discomfort")
        else:
            st.error("There is an acute chance of heart attack.Please visit a doctor immediately")

def about():
    st.title("About")
    st.subheader("This application analyzes a heart attack dataset available in kaggle. The data is then processed and various types of classification algorithms namely Logistic Regression ,Decision tree,K nearesr neighbour ,Svc are applied.The one with the best result is taken.")
    st.header("Having Problem in some terms ? Look here...")
    st.subheader("1.Typical angina is the discomfort that is noted when the heart does not get enough blood or oxygen. Typically, this is caused by blockage or plaque buildup in the coronary arteries")
    st.subheader("2.People with atypical chest pain experience symptoms that are rather similar to gastrointestinal, respiratory and musculoskeletal diseases.")
    st.subheader("3.Slope :The ST segment shift relative to exercise-induced increments in heart rate, the ST/heart rate slope (ST/HR slope), has been proposed as a more accurate ECG criterion for diagnosing significant coronary artery disease (CAD).")
    st.subheader("4.thal rate - 2 = normal; 1 = fixed defect; 3 = reversable defect")
    st.subheader("5.Blood vessels - number of major vessels (0-3) colored by flourosopy")
    st.info("Heart attack and Heart related diseases have been rising in India and throughout the world.Its time we become aware of it and predict it before its too late...")

def plot():
    st.title("Plot and See")
    graph=st.selectbox("See how different parameters relate to heart diseases",["Age","Cholesterol","Sex"])
    if(graph=="Age"):
        st.line_chart(df['age'])
    if(graph=="Cholesterol"):
        st.line_chart(df['chol'])
    if(graph=="Sex"):
        fig, ax = plt.subplots()
        count=[93,72]
        name=['Male','Female']
        ax.pie(count,labels=name)
        st.pyplot(fig)

def meet():
    st.title("Meet Me")
    st.image("data//souvik1.png",width=300)
    st.subheader('Hii ! This is Souvik')
    st.subheader('I am enthusiastic about machine learning and data science.Afterall whats learning if its not for a cause ! My cause is to make projects that help the society solve real world problems...')
    st.write("Reach me @ [LinkedIn](https://www.linkedin.com/in/souvik-ghosh-3b8b411b2/)")
    st.write("Gmail-souvikg544@gmail.com")

df=pd.read_csv('data//heart.csv')
df_pos=df[df['output']==1]
df_pos['sex'] = df_pos['sex'].replace([0],'Female')
df_pos['sex'] = df_pos['sex'].replace([1],'Male')

nav=st.sidebar.radio("Know hows your Heart",["Home","Plot","About The Project","Meet me"])

if (nav =="Home"):
    hometab()
if(nav=="Meet me"):
    meet()
if(nav == "About The Project"):
    about()
if(nav=="Plot"):
    plot()
    

    

















    

    


