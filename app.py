from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('rf_model.pkl')
encoder = joblib.load('encoder.pkl')

def load_columns(filename):
    with open(filename, 'r') as f:
        columns = f.readlines()
    return [col.strip() for col in columns]

def recommend_top_yields_corrected(month, year, model, encoder):

    yields = ["Corn", "Tomato", "Onion", "Pumpkin", "Potato", "Ginger", "Garlic", "Radish", 
              "Leak", "Chilies", "Pepper", "Cowpea", "Cocoa", "Cardamon"]

    data = {
        'Month': [month] * len(yields),
        'Year': [year] * len(yields)
    }

    x_train_columns = load_columns('x_train_columns.txt')

    for column in x_train_columns[2:]:
        yield_type = column.split("_")[1]
        data[column] = [1 if yield_type == y else 0 for y in yields]
    df_input = pd.DataFrame(data)
    
    predicted_scores = model.predict(df_input)
    
    top_yields = np.array(yields)[np.argsort(predicted_scores)[::-1][:5]]
    
    return list(top_yields)


@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json(force=True)
    month = data['month']
    year = data['year']
    
    top_yields = recommend_top_yields_corrected(month, year, model, encoder)
    
    return jsonify(top_yields)


if __name__ == '__main__':
    app.run(port=5000, debug=True)