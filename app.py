from flask import Flask, request, render_template
import pandas as pd
import joblib
from DialogueMapper import DialogueMapper
import settings


# Declare a Flask app
app = Flask(__name__)
model = joblib.load(settings.MODEL)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Get values through input bars
        line = request.form.get("line")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[line]], columns = ["line"])
        
        # Get prediction
        preds = model.predict(X)[0]
        
        return render_template("website.html", output=preds)

    else:
        return render_template("website.html", output="*Predicted character*")
        

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
