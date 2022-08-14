# Heart Attack Prediction Web App

![istockphoto-1128931450-612x612](https://user-images.githubusercontent.com/63863911/184526983-66353672-a1db-4729-b54d-0991b7de1e69.jpg)




## Backend
- A decision tree classifier is used over a set of different 14 parameters of an ecg report to estimate whether the patient has probability of a possible heart attack in the coming days
- Intense data processing involving outlier removal and future importance is done to make the model more robust.
- The data is analyzed to get the statistics of heart attack over gender,age, region and blood groups.
- The model is trained using griddsearch for the optimal set of hyperparameters to achieve an accuracy of 81.25 % of the test set.

## Frontend
Thanks to Streamlit for their wonderful platform allowing data enthusiasts to show their work with ease.

### The app is deployed on heroku. Check it out here - 


[https://heart-attack-prediction-india.herokuapp.com/](https://heart-attack-prediction-india.herokuapp.com/)

## Futture work
- The textual input will be automated to read inputs directly from the ecg report images.
- The incoming data will be stored in the database and added to our dataset for retraining ouyr model after every instance.
