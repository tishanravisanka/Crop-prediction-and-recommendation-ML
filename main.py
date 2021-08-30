from flask import Flask,request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

reponse = []
req = []

def predict(v1,v2,v3,v4,v5,v6,v7):
        
    df = pd.read_csv('crop_recommendation.csv')
    features = df[['N','P','K','temperature','humidity','ph','rainfall']]
    target = df['label']
    labels = df['label']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size=0.2, random_state=2)

    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)

    # save the model to disk
    filename = 'RandomForest_model.sav'
    pickle.dump(RF, open(filename, 'wb'))

    # RF = pickle.load(open("RandomForest_model.sav", 'rb'))

    data = np.array([[v1,v2, v3, v4, v5, v6, v7]])
    prediction = RF.predict(data)

    return (prediction[0])
    
# predict
@app.route('/predict',methods = ['POST', 'GET'])
def result():
    area = request.args.get("area")
    province = request.args.get("province")
    return predict(area,province),200


# recommend
@app.route('/recommend',methods = ['POST', 'GET'])
def recommendResult():

    # data = predict(nitrogen,phosphorus, potassium, temperature, humidity, ph, rainfall)
    # print(data)
    #  return predict(nitrogen,phosphorus, potassium, temperature, humidity, ph, rainfall),200
    return "data",200
    # return predict(90,48, 40, 28.603016, 85.3, 10.7, 199.91),200

if __name__ =="__main__":
    app.run(debug=True)

