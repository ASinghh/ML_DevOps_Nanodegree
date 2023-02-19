'''
A library of functions to find customers who are likely to churn

Author: Ashutosh Singh

Date: 12th of February, 2023
'''

# Import Packages

import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def create_dir_struct(
    dir_list=[
        "images",
        "images/eda",
        "images/results",
        "logs",
        "models",
        "dataset_characteristics"]):
    '''
    This function checks for required directory structure and creates the
    directories that are missing.

    Adding dataset_characteristics to save outputs from the df.head(),
    df.shape, and  df.isnull.sum(), and df.describe().

    input:
            dir_list: list of directory names
    output:
            None
    '''
    for dirct in dir_list:
        path = r"./" + dirct
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            continue


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Create the required directory strucutre
    create_dir_struct()

    # Save dataset characteristics to files for manual viewing
    df.head().to_csv(r'./dataset_characteristics/df_head')
    np.savetxt(r'./dataset_characteristics/df_shape', df.shape, fmt='%d')
    df.isnull().sum().to_csv(r'./dataset_characteristics/df_null_counts')
    df.describe().to_csv(r'./dataset_characteristics/df_description')

    # Create binary variable for Churn
    if 'Churn' not in df:
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

    # Create and save the binary churn distribution
    plt.figure(figsize=(20, 10))
    df['Attrition_Flag'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.jpeg')

    # Create and save customer age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    # Create and save marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    # Create and save total transaction distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(fname='./images/eda/total_transaction_distribution.png')

    #  Create and save correlation heat map
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name.
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for analysis
    '''

    if 'Churn' not in df:
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        
    for category in category_lst:
        column_lst = []
        column_groups = df.groupby(category).mean()['Churn']

        for val in df[category]:
            column_lst.append(column_groups.loc[val])

        if response:
            df[category + '_' + response] = column_lst
        else:
            df[category] = column_lst
    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name.
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Creating binary variable 'Chrun', for cases when EDA is not performed before hand
    if 'Churn' not in df:
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

    # categorical features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    # feature engineering
    df = encoder_helper(df, cat_columns, response)

    # target feature
    y = df['Churn']

    # Create dataframe
    X = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # Features DataFrame
    X[keep_cols] = df[keep_cols]

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds, model_name, result_dir):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder. Changing parameters to make more modular.
    input:
            y_train      : Pandas Dataframe- training response values
            y_test       : Pandas Dataframe- test response values
            y_train_preds: Pandas Dataframe- training predictions from the fitted model
            y_test_preds : Pandas Dataframe- test predictions from the fitted model
            model_name   : str- Name of the classifier
            result_dir   : str- Path of dir to store report image

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str(model_name + ' Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(model_name + ' Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')

    result_file = result_dir + 'classf_rep_' + model_name  + '.png'
    plt.savefig(fname=result_file)


def feature_importance_plot(model_file, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model     : model object containing feature_importances_
            X_data    : pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Putting a test to check if the clf has feature importance feature
    try:
        # Loading model
        model_obj = joblib.load(model_file)
        model_name = (model_file.split("/")[-1]).split(".")[0]

        # Feature importances
        importances = model_obj.feature_importances_

        # Sort Feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(25, 15))

        # Create plot title
        plt.title(model_name + " Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        # Save the image
        plt.savefig(fname=output_pth + model_name + '_feature_importances.png')

    except AttributeError:
        pass


def roc_plotter(model, X_test, y_test, roc_file):
    '''
    Function to draw the ROC graph and saving it to a file.
    input:
            model   : model object containing feature_importances_
            X_test  : pandas dataframe of test feature set
            y_test  : pandas dataframe of test target variables
            roc_file: file to save the plot at

    output:
             None
    '''
    plt.figure(figsize=(15, 8))
    plot_roc_curve(model, X_test, y_test)
    plt.savefig(fname=roc_file)
    plt.close()


def train_models(df, model_dir, result_dir, clf, response_var_name, param_grid=None):
    '''
    train, store model results: images + scores, and store models. Adding more function parameters
    to make this function more modular
    input:
              X_train    : Pandas Dataframe- X training data
              X_test     : Pandas Dataframe- X testing data
              y_train    : Pandas Dataframe- y training data
              y_test     : Pandas Dataframe- y testing data
              model_dir  : str- Path of dir to save final model
              result_dir : str- Path of dir to save results
              clf        : SkLearn Classifier
              param_grid : {str(hyper_param:[value]}- Parameter grid for grid search
    output:
              None
    '''
    # Creating directory structure for cases when EDA is not performed
    create_dir_struct()

    # Seperating Features and target variable, and performing train-test split
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, response=response_var_name)

    # Checking for parameter grid, preforming grid search if the grid is provided
    # and fitting the classifier
    if param_grid is not None:
        model = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        model.fit(X_train, y_train)
        model = model.best_estimator_

    else:
        model = clf.fit(X_train, y_train)

    # Performing inference on feature sets
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    # Generating model file name
    model_name = str(clf)
    model_file = model_dir + model_name + '.pkl'
    roc_file = result_dir + 'roc_curve_' + model_name + '.png'

    # Saving the best model
    joblib.dump(model, model_file)

    # Plotting and saving the ROC Curve
    roc_plotter(model, X_test, y_test, roc_file)

    # Creating and saving the classification_report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds, model_name, result_dir)

    # Creating and saving the Feature Importance plot(for classifiers that
    # support this option)
    feature_importance_plot(model_file, X_test, result_dir)


if __name__ == '__main__':
    # Import data
    BANK_DF = import_data(pth='./data/bank_data.csv')

    # Perform EDA
    perform_eda(df=BANK_DF)

    # Initializing (classifier, parameter grid) tuple for sample runs
    clf_paramGrid = [
        (RandomForestClassifier(
            random_state=42), {
            'n_estimators': [
                200, 500], 'max_features': [
                    'auto', 'sqrt'], 'max_depth': [
                        4, 5, 100], 'criterion':[
                            'gini', 'entropy']}), (LogisticRegression(
                                solver='lbfgs', max_iter=3000), None)]

    # Initializing directories to save models and results
    MODEL_DIR = r'./models/'
    RESULT_DIR = r'./images/results/'

    # Sample runs
    for classif, grid in clf_paramGrid:
        train_models(
            df=BANK_DF,
            model_dir=MODEL_DIR,
            result_dir=RESULT_DIR,
            clf=classif,
            response_var_name = 'Churn',
            param_grid=grid)