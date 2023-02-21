'''
A module for unit tests of the functions defined in churn_library.py

Author: Ashutosh Singh

Date: 19th of February, 2023
'''


import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: Success")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_perform_eda():
    '''
    test perform eda function
    '''
    dataframe = cls.import_data("./data/bank_data.csv")

    # Test perform_eda function run
    try:
        cls.perform_eda(dataframe)
        logging.info("EDA Function Run: Success")
    except BaseException as err:
        logging.error("EDA Function Run: Error")
        raise err

    image_list = [
        "churn_distribution.jpeg",
        "customer_age_distribution.png",
        "marital_status_distribution.png",
        "total_transaction_distribution.png",
        "heatmap.png"]
    file_list = ["df_description", "df_head", "df_null_counts", "df_shape"]

    # Testing the individual EDA images for successful creation
    for image in image_list:
        try:
            assert os.path.isfile(r'./images/eda/' + image) is True
            logging.info('%s file was generated: Success', image)
        except AssertionError as err:
            logging.error('%s file was not generated: Error', image)
            raise err

    # Testing the individual data characteristics files for successful creation
    for file in file_list:
        try:
            assert os.path.isfile(r'./dataset_characteristics/' + file) is True
            logging.info('%s file was generated: Success', file)
        except AssertionError as err:
            logging.error('%s file was not generated: Error', file)
            raise err


def test_encoder_helper():
    '''
    Test encoder_helper() function from the churn_library module
    '''

    dataframe = cls.import_data("./data/bank_data.csv")
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    # testing for both cases, response name provided by user and left blank
    response_list = [None, "churn"]

    for i, response in enumerate(response_list):
        try:
            df = cls.encoder_helper(dataframe.copy(
                "deep"), cat_columns, response)
            assert df.shape == (
                dataframe.shape[0],
                dataframe.shape[1] +
                1 +
                i *
                len(cat_columns))
            logging.info(
                "Testing encoder_helper for response == %s: Success", response)
        except AssertionError as err:
            logging.error(
                "Testing encoder_helper for response == %s: Error", response)
            raise err


def test_perform_feature_engineering():
    '''
    Test perform_feature_engineering() function from the churn_library module
    '''
    # Load the DataFrame
    dataframe = cls.import_data("./data/bank_data.csv")

    (X_train, X_test, y_train, y_test) = cls.perform_feature_engineering(
        df=dataframe,
        response='Churn')
    
    # Assert the creation of target variable
    try:
        assert 'Churn' in dataframe.columns
        logging.info("Target variable created: Success")
    except KeyError as err:
        logging.error('Target variable was not created: Error')
        raise err
    
    # Assert test and train are 30 and 70 % of the df
    try:
        assert (((X_test.shape[0] >= dataframe.shape[0] * 0.3 - 1) or
                 (X_test.shape[0] >= dataframe.shape[0] * 0.3 + 1)) and
                (X_train.shape[0] == dataframe.shape[0] - X_test.shape[0]))
        logging.info(
            'Test-Train split was performed in the right ratios: Success')
    except AssertionError as err:
        logging.error(
            'Test-Train split was not performed in the right ratios: Error')
        raise err
    
    # Assert the row numers of y and X dataframes are the same
    try:
        assert (X_test.shape[0] == y_test.shape[0]) and\
            (X_train.shape[0] == y_train.shape[0])
        logging.info(
            'Target variable dataframes are the right shapes: Success')
    except AssertionError as err:
        logging.error(
            'Target variable dataframes are not the right shapes: Failure')
        raise err


def test_train_models():
    '''
    Test train_models() function from the churn_library module
    '''
    # Load the DataFrame
    dataframe = cls.import_data("./data/bank_data.csv")

    # Directory
    model_dir = r'./models/'
    result_dir = r'./images/results/'
    
    # Initialize list of classifiers and the expected plot prefixes
    class_list = [LogisticRegression(), RandomForestClassifier()]
    plot_list = ['roc_curve_', 'classf_rep_']

    for classif in class_list:
        # Assert the model is trained
        try:
            cls.train_models(
                df=dataframe,
                model_dir=model_dir,
                result_dir=result_dir,
                clf=classif,
                response_var_name='Churn',
                param_grid=None)
            logging.info(
                '%s Model was trained successfuly: Success', str(classif)[0:50] + ")")
        except BaseException as err:
            logging.error(
                '%s Model was not trained successfuly: Failure', str(classif)[0:50] + ")")
            raise err

        # Assert the model file is saved
        try:
            assert os.path.isfile(model_dir + str(classif)[0:50] + ")" + '.pkl') is True
            logging.info('Model File %s was found: Success',
                         str(classif)[0:50] + ")" + '.pkl')
        except AssertionError as err:
            logging.error('Model File %s was not found: Failure',
                          str(classif)[0:50] + ")" + '.pkl')
            raise err

        # Assert the creation of ROC and Characterisitcs plots
        for plot in plot_list:
            try:
                assert os.path.isfile(
                    result_dir + plot + str(classif)[0:50] + ")" + '.png') is True
                logging.info(plot + 'for %s was found: Success', str(classif)[0:50] + ")")
            except AssertionError as err:
                logging.error(
                    plot + 'for %s was not found: failure', str(classif)[0:50] + ")")
                raise err


if __name__ == "__main__":
    test_import()
    test_perform_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
