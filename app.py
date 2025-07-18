from flask import Flask,render_template,request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
app=Flask(__name__)
data=pd.read_csv('Crop_recommendation.csv')
X=data.drop('label',axis=1)
y=data['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(X,y)
fields=['N','P','K','temperature','humidity','ph','rainfall']
@app.route('/',methods=['GET'])
def index():
    return render_template("index.html",fields=['N','P','K','temperature','humidity','ph','rainfall'])
@app.route('/predict',methods=['POST'])
def predict():
    input_vals=[float(request.form[f])for f in fields]
    prediction=model.predict([input_vals])[0]
    probs=model.predict_proba([input_vals])[0]
    crops=model.classes_
    plt.figure(figsize=(10,5))
    plt.bar(crops,probs,color='green')
    plt.xticks(rotation=90)
    plt.title('Crop Prediction Probabilities')
    plt.xlabel('Crops')
    plt.ylabel('Probaility')
    plt.tight_layout()
    plt.savefig("static/chart.png")
    plt.close()
    return render_template("result.html",prediction=prediction)
if __name__=="__main__":
    app.run(debug=True)