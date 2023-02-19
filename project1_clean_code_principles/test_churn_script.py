import os
import logging
from churn_library import  import_data, perform_eda

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: Success")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    dataframe = import_data("./data/bank_data.csv")
    try:
        perform_eda(dataframe)
        logging.info("EDA Function Run: Success")
    except BaseException:
        logging.error("EDA Function Run: Error")

    image_list = ["churn_distribution.jpeg","customer_age_distribution.png",
                  "marital_status_distribution.png", "total_transaction_distribution.png",
                  "heatmap.png"]
    
    file_list = ["df_description", "df_head", "df_null_counts", "df_shape"]

    # Testing the individual EDA images for successful creation
    for image in image_list:
        try:
            assert os.path.isfile(r'./images/eda/' + image) is True
            logging.info('%s file was generated: Success', image)
        except AssertionError as err:
            logging.error('%s file was not generated: Error', image)
            
    # Testing the individual data characteristics files for successful creation     
    for file in file_list:
        try:
            assert os.path.isfile(r'./dataset_characteristics/' + file) is True
            logging.info('%s file was generated: Success', file)
        except AssertionError as err:
            logging.error('%s file was not generated: Error', file)





if __name__ == "__main__":
    test_import()
    test_eda()





