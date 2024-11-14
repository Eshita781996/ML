import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
#model = pickle.load(open("model.pkl", "rb"))



@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/home")
def GotoHome():
    return render_template("index.html")

@flask_app.route("/houseprediction")
def houseprediction():
    return render_template("houseprediction.html")

@flask_app.route("/predictprice", methods = ["POST"])
def predictprice():
    model = pickle.load(open("linearregression.pickle", "rb"))  
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    y_pred = model.predict(features)
    return render_template("houseprediction.html", prediction_text = "The price of the house is {}. ".format(round(y_pred[0])) )

if __name__ == "__main__":
    flask_app.run(debug=True)
