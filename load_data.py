import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    # Example dataset (replace with your real dataset)
    data = {
        'Location': ['Downtown', 'Suburbs', 'Downtown', 'Rural', 'Suburbs', 'Suburbs', 'Downtown', 'Rural'],
        'Area': [1000, 1500, 1200, 1800, 1300, 1250, 1100, 2000],
        'Amenities': [5, 4, 6, 3, 7, 5, 4, 8],
        'Price': [500000, 350000, 450000, 250000, 400000, 380000, 420000, 300000]
    }

    # Convert into a pandas DataFrame
    df = pd.DataFrame(data)

    """df.head() will display the first 5 rows of the DataFrame.
    If you want to see a different number of rows, you can pass an integer to df.head(n), 
    where n is the number of rows you want to view."""
    # Check first few rows of the data
    print(df.head())

    # One-hot encode the 'Location' column
    """pd.get_dummies():

    This is a function provided by pandas that automatically creates one-hot encoded variables for categorical columns in a DataFrame.
    One-hot encoding transforms categorical variables into a format that is better suited for machine learning models,
     by creating a separate binary (0 or 1) column for each possible category within the original categorical column.
    columns=['Location']:
    
    This argument specifies that we want to apply the one-hot encoding to the Location column only.
    Location is a categorical column (containing values like 'Downtown', 'Suburbs', etc.).
    drop_first=True:
    
    By default, when we apply one-hot encoding, pandas creates a separate column for each possible category in the
    column. For example, if there are 3 unique values in the Location column, pandas would create
    3 new columns — one for each category (e.g., Location_Downtown, Location_Suburbs, Location_Rural).
    However, the value drop_first=True tells pandas to drop the first column of the resulting one-hot encoded columns.
    This helps to avoid multicollinearity (when features are highly correlated with each other), especially in linear 
    models. This is a common practice in machine learning, as the remaining columns will capture the necessary information."""
    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

    # Check the processed data
    print(df.head())

    # Split the dataset into features and target
    X = df.drop('Price', axis=1)  # Features axis=1 indicates that we're dropping a column (if you were dropping rows, you would use axis=0).
    y = df['Price']  # Target variable

    # Split the dataset into training and testing sets. This is a crucial step in machine learning to evaluate how well your model performs on unseen data.
    """Inputs to the function:

    X: The feature set (independent variables) that the model will use as inputs.
    y: The target variable (dependent variable) that the model is trying to predict.
    test_size=0.2:
    
    Specifies the proportion of the data to be used as the testing set. Here, 0.2 means 20% of the dataset will be used for testing, and the remaining 80% will be used for training.
    Example: If you have 100 rows of data, 80 will go to the training set, and 20 will go to the testing set.
    random_state=42:
    
    Ensures that the random splitting of data into training and testing sets is reproducible. Setting random_state to a specific value (like 42) ensures that the same split will happen every time the code is run.
    If random_state is not specified or left as None, the function will randomly split the data every time, which might lead to different results each time you run the code.
    Outputs of the function:
    
    X_train, X_test, y_train, y_test
    X_train: The feature set for the training dataset (80% of X).
    X_test: The feature set for the testing dataset (20% of X).
    y_train: The target variable for the training dataset (80% of y).
    y_test: The target variable for the testing dataset (20% of y)."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the shape of the splits
    print(f'Training Features: {X_train.shape}, Test Features: {X_test.shape}')


    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")

    # Plot actual vs predicted prices
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.show()


    # Get feature importances
    # importances = rf_model.feature_importances_
    # features = X.columns
    #
    # # Create a bar plot for feature importances
    # plt.figure(figsize=(10,6))
    # sns.barplot(x=importances, y=features)
    # plt.title("Feature Importance")
    # plt.show()


