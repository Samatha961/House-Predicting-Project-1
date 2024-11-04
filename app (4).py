from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create the Flask app
app = Flask(__name__)

# Load your dataset (replace with your actual dataset)
data = {
    'Square_Footage': [1360, 4272, 3592, 966, 4926, 3944],  # Add all your data here
    'Num_Bedrooms': [2, 3, 1, 1, 2, 5],
    'Num_Bathrooms': [1, 3, 2, 2, 1, 3],
    'Year_Built': [1981, 2016, 2016, 1977, 1993, 1990],
    'Lot_Size': [0.59963664, 4.753013849, 3.63482272, 2.730666876, 4.699072555, 2.475930044],
    'Garage_Size': [0, 1, 0, 1, 0, 2],
    'House_Price': [262382.8523, 985260.8545, 777977.3901, 229698.9187, 1041740.859, 879796.9835]
}

df = pd.DataFrame(data)

# Train a simple linear regression model
X = df[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size']]
y = df['House_Price']
model = LinearRegression()
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        square_footage = float(request.form['square_footage'])
        num_bedrooms = int(request.form['num_bedrooms'])
        num_bathrooms = int(request.form['num_bathrooms'])
        year_built = int(request.form['year_built'])
        lot_size = float(request.form['lot_size'])
        garage_size = int(request.form['garage_size'])

        # Prepare input for the model
        input_data = np.array([[square_footage, num_bedrooms, num_bathrooms, year_built, lot_size, garage_size]])
        prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
