from flask import Flask, request, render_template
import pandas as pd
import joblib
from GrantClassifier import GrantClassifier


# Declare a Flask app
app = Flask(__name__)
model = joblib.load("models/grantclf_xgb_05-02-2022.pkl")

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Get values through input bars
        zweck = request.form.get("zweck")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[zweck]], columns = ["Zweck"])
        
        # Get prediction
        preds = model.predict(X)[0]
        
        return render_template("website.html", output=preds)

    else:
        return render_template("website.html", output="*Predicted purpose*")
        

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
