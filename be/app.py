from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

# Global dictionary to store the data
data_map = {}

@app.route("/satisfaction", methods=['POST'])
def addEmployeeData():
    try:
        # Get the JSON data from the request body
        data = request.get_json()
        data_map[data['employeeNumber']] = data
        print (data_map)
        return jsonify(data_map)
    except Exception as e:
        return jsonify({'error': 'Invalid JSON payload' + e}), 400

@app.route("/satisfaction/<employeeNumber>", methods=['GET'])
def getEmployeeData(employeeNumber):
    try:
        if employeeNumber in data_map:
            # Call Function
            print ("Req recieved", employeeNumber)
            res = predict_employee_depression_level(data_map[employeeNumber])
            response = {"depressionStatus": res}
            return jsonify(response), 200
        else:
            return jsonify({'error': 'Employee not found.'}), 404
    except Exception as e:
        return jsonify({'error': 'Error retrieving data' + str(e)}), 500

def createBoxPlotsForColumns(df):
    sns.boxplot(x='Age', data=df)  # No outliers here
    sns.boxplot(x='DailyRate', data=df)  # No outliers here
    sns.boxplot(x='DistanceFromHome', data=df)  # No outliers here
    sns.boxplot(x='EmployeeNumber', data=df)  # No outliers here
    sns.boxplot(x='HourlyRate', data=df)  # No outliers here
    sns.boxplot(x='MonthlyIncome', data=df)  # No outliers here
    sns.boxplot(x='MonthlyRate', data=df)  # No outliers here
    sns.boxplot(x='PercentSalaryHike', data=df)  # No outliers here

def makeCategoricalValuesNumeric(df, column):
    unique_values = df[column].unique()

    # Create a mapping dictionary
    mapping = dict(zip(unique_values, range(len(unique_values))))

    # Replace the categorical values with numbers
    df[column] = df[column].replace(mapping)

def convertCategoricalValuesNumeric(df):
    makeCategoricalValuesNumeric(df, 'Gender')
    makeCategoricalValuesNumeric(df, 'Attrition')
    makeCategoricalValuesNumeric(df, 'BusinessTravel')
    makeCategoricalValuesNumeric(df, 'Department')
    makeCategoricalValuesNumeric(df, 'EducationField')
    makeCategoricalValuesNumeric(df, 'JobRole')
    makeCategoricalValuesNumeric(df, 'MaritalStatus')
    makeCategoricalValuesNumeric(df, 'Over18')
    makeCategoricalValuesNumeric(df, 'OverTime')

def linearRegression(df_pca):
    X_train, X_test, y_train, y_test = train_test_split(df_pca.drop('DepressionLevel', axis=1),
                                                        df_pca['DepressionLevel'], test_size=0.2,
                                                        random_state=42)
    # Create a linear regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Use the trained model to make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the performance of the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print('Mean squared error:', mse)

    # Set threshold to convert continuous predictions to classes
    threshold = 2

    # Convert predictions to classes based on threshold
    y_pred_classes = (y_pred > threshold).astype(int)
    # Convert test to classes based on threshold
    y_test_classes = (y_test > threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print(cm)
    # Accuracy score
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print("Linear regression confusion matrix:\n", cm)
    print("linear regression Accuracy score:", accuracy)

def logisticRegression(df_pca):
    X_train, X_test, y_train, y_test = train_test_split(df_pca.drop('DepressionLevel', axis=1),
                                                        df_pca['DepressionLevel'], test_size=0.2,
                                                        random_state=42)

    # Create a logistic regression model
    model = LogisticRegression()

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Set threshold to convert continuous predictions to classes
    threshold = 2

    # Convert predictions to classes based on threshold
    y_pred_classes = (y_pred > threshold).astype(int)
    # Convert test to classes based on threshold
    y_test_classes = (y_test > threshold).astype(int)

    # Evaluate the performance of the model
    confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)

    print("Logistic regression confusion matrix:\n", confusion_mat)
    print("Logistic regression Accuracy score:", accuracy)

def knn(df_pca):
    X_train, X_test, y_train, y_test = train_test_split(df_pca.drop('DepressionLevel', axis=1),
                                                        df_pca['DepressionLevel'], test_size=0.2,
                                                        random_state=42)
    # Creating the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)

    # Fitting the model on the training data
    knn.fit(X_train, y_train)

    # Predicting the class labels on the testing data
    y_pred = knn.predict(X_test)

    # Set threshold to convert continuous predictions to classes
    threshold = 2

    # Convert predictions to classes based on threshold
    y_pred_classes = (y_pred > threshold).astype(int)
    # Convert test to classes based on threshold
    y_test_classes = (y_test > threshold).astype(int)

    # Evaluate the performance of the model
    confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print("KNN confusion matrix:\n", confusion_mat)
    print("KNN Accuracy score:", accuracy)

def NaiveBayes(df_pca):
    X_train, X_test, y_train, y_test = train_test_split(df_pca.drop('DepressionLevel', axis=1),
                                                        df_pca['DepressionLevel'], test_size=0.2,
                                                        random_state=42)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    # Set threshold to convert continuous predictions to classes
    threshold = 2

    # Convert predictions to classes based on threshold
    y_pred_classes = (y_pred > threshold).astype(int)
    # Convert test to classes based on threshold
    y_test_classes = (y_test > threshold).astype(int)

    # Evaluate the performance of the model
    confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)

    print("Naive Bayes confusion matrix:\n", confusion_mat)
    print("Naive Bayes Accuracy score:", accuracy)

def randomForest(df_pca):
    X_train, X_test, y_train, y_test = train_test_split(df_pca.drop('DepressionLevel', axis=1),
                                                        df_pca['DepressionLevel'], test_size=0.2,
                                                        random_state=42)

    # Build the random forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf_model.predict(X_test)

    # Set threshold to convert continuous predictions to classes
    threshold = 2

    # Convert predictions to classes based on threshold
    y_pred_classes = (y_pred > threshold).astype(int)
    # Convert test to classes based on threshold
    y_test_classes = (y_test > threshold).astype(int)

    # Evaluate the performance of the model
    confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print('Random Forest Confusion Matrix:\n', confusion_mat)
    print('Random Forest Accuracy:', accuracy)

def buildModel():
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

    # Calculate the DepressionLevel based on the equation
    df['DepressionLevel'] = 4.0278 - 0.3426 * df['JobSatisfaction']

    # Map the DepressionLevel to the range of 1 to 5 using pandas.cut()
    bins = [float('-inf'), 2.314, 2.75, 3.186, 3.622, float('inf')]
    labels = [1, 2, 3, 4, 5]
    df['DepressionLevel'] = pd.cut(df['DepressionLevel'], bins=bins, labels=labels)

    # Preprocessing
    # Print the first few rows of the updated dataset
    print(df.head(30))

    # Check for missing values
    print(df.isnull().sum())  # No null values in the dataset

    # Check for duplicates
    print('Duplicated Values:', df.duplicated().sum())  # No duplicates in the dataset

    # Create box plots
    createBoxPlotsForColumns(df)
    plt.show()

    # Create a Standard Scaler object
    scaler = StandardScaler()

    # Fit the scaler to the data and transform the Age and MonthlyIncome columns
    df[['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeNumber', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'StandardHours', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']] = scaler.fit_transform(df[['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeNumber', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'StandardHours', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']])
    print('Scaled')

    # Convert categorical variables into numbers / Encoding
    convertCategoricalValuesNumeric(df)

    # Feature selection:
    # Dropping EmployeeCount column
    df = df.drop('EmployeeCount', axis=1)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df)
    # create a new DataFrame with the reduced feature matrix
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    # add the target variable to the new DataFrame
    df_pca['DepressionLevel'] = df['DepressionLevel']
    sns.scatterplot(x='PC1', y='PC2', hue='DepressionLevel', data=df_pca)
    plt.show()

    linearRegression(df_pca)
    logisticRegression(df_pca)
    knn(df_pca)
    NaiveBayes(df_pca)
    randomForest(df_pca)

def preprocess_input_data(df):
    # Calculate the DepressionLevel based on the equation
    df['DepressionLevel'] = 4.0278 - 0.3426 * df['JobSatisfaction']

    # Map the DepressionLevel to the range of 1 to 5 using pandas.cut()
    bins = [float('-inf'), 2.314, 2.75, 3.186, 3.622, float('inf')]
    labels = [1, 2, 3, 4, 5]
    df['DepressionLevel'] = pd.cut(df['DepressionLevel'], bins=bins, labels=labels)

    # Create a Standard Scaler object
    scaler = StandardScaler()

    # Fit the scaler to the data and transform the Age and MonthlyIncome columns
    df[['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeNumber', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
        'PercentSalaryHike', 'StandardHours', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsWithCurrManager']] = scaler.fit_transform(df[['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeNumber',
                                                            'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
                                                            'PercentSalaryHike', 'StandardHours', 'TotalWorkingYears',
                                                            'YearsAtCompany', 'YearsInCurrentRole',
                                                            'YearsWithCurrManager']])

    # Convert categorical variables into numbers / Encoding
    # df = pd.get_dummies(df, columns=['Gender', 'Attrition', 'BusinessTravel', 'Department', 'EducationField',
    #                                  'JobRole', 'MaritalStatus', 'Over18', 'OverTime'])
    convertCategoricalValuesNumeric(df)

    # Feature selection:
    # Dropping EmployeeCount column
    df = df.drop('EmployeeCount', axis=1)

    return df

def predict_employee_depression_level(employee_data):
    # Load the dataset and preprocess the data
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    testElemnt = df.iloc[2]
    df = preprocess_input_data(df)

    print("Hello", testElemnt)

    # Create PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df)

    # Create a new DataFrame with the reduced feature matrix
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    # Add the target variable to the new DataFrame
    df_pca['DepressionLevel'] = df['DepressionLevel']

    # Create a linear regression model
    model = LinearRegression()

    # Split the data into features and target variable
    X = df_pca.drop('DepressionLevel', axis=1)
    y = df_pca['DepressionLevel']

    # Train the model using the entire dataset
    model.fit(X, y)

    # Preprocess the provided employee data
    employee_df = pd.DataFrame(testElemnt).T
    employee_df = preprocess_input_data(employee_df)

    # Perform PCA on the employee data
    X_employee_pca = pca.transform(employee_df)

    # Predict the employee's depression level
    employee_depression_level = model.predict(X_employee_pca)[0]

    return employee_depression_level
