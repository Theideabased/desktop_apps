import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
from pathlib import Path

data_csv = Path(__file__).parents[1]  /'smart_collector/car_insurance.csv'
data = pd.read_csv(data_csv)

features = ['Manufacturer','Model', 'Category','Mileage','Prod. year','Engine volume', 'Airbags']
# get your features
data_used = data[features]

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in data_used.columns if data_used[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in data_used.columns if data_used[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model
my_pipeline.fit(data_used, data[['Price']])

# Define the app
def app():
    st.title('Smart collector')
    st.write('input your car details and get your policy:')
    
    # Create input widgets for each feature
    manufacturer = st.selectbox("manufacturer", options=(list( data['Manufacturer'].unique())))
    model = st.selectbox("Model", options=(list( data['Model'].unique())))
    mileage = st.number_input('Mileage')
    engine_volume = st.slider('Engine Volume', min_value=1, max_value=6, step=1)
    airbags = st.slider('airbags', min_value=1, max_value=15, step=1)
    category = st.selectbox("Category", options =(data['Category'].unique()))
    prod_yr = st.selectbox("production year", options=(list( data['Prod. year'].unique())))
    # set your values
    df_from_input = pd.DataFrame([{
   'Manufacturer' : manufacturer,
   'Model': model,
   'Category': category,
   'Mileage': mileage,
   'Prod. year': prod_yr,
   'Engine volume': engine_volume,
   'Airbags': airbags,
  }])

    #price = model.predict(df_from_input)
    #return price

    # Display the predicted price to the user
    if st.button('Submit'):
        price = my_pipeline.predict(df_from_input)
        st.success(f'your policy is {price[0]:,.2f} naira.')
        if price >= 100000:
            st.write('this vechicle has a high risk and it is unacceptable')
        elif price >= 50000:
            st.write('this vechicle has a high risk but it is acceptable')
        else:
            st.write('this vechicle has a low risk')

    
if __name__ == '__main__':
    app()
