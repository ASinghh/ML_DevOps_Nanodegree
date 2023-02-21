# Predict Customer Churn

Project **Predict Customer Churn** from **Clean Coding Principles** module of Udacity's ML DevOps Engineer Nanodegree course.

## Project Description

The Churn_notebook was provided by Udacity as is in the repo. The first part of the project was to define a module (churn_library.py) to modularize the methods used in the notebook, into functions. The module is expected to follow the clean coding principles that were tought in the course.
The second part of the project was to write unit tests for the functions defined in churn_library.py (in churn_script_logging_and_tests.py) and log the test resutls.

## Project Directory Structure

The structure of this project directory tree is displayed as follows:

```
├── data
│   └── bank_data.csv (Data file)
├── dataset_characteristics (Features of the dataframe
│   │   ├── df_description
│   │   ├── df_head
│   │   ├── .
│   │   └── .
├── images
│   ├── eda (Plots generated during EDA)
│   │   ├── churn_distribution.jpeg
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   │   └── .
│   └── results (model results)
│       ├── LogisticRegression(max_iter=3000)_classf_rep.png
│       ├── .
│       ├── .
│       └── .
├── logs (Unit test logs)
│   └── churn_library.log
├── models (Stored model as pkl files)
│   ├── ....pkl
│   └── .
├── Guide.ipynb (Directions provided by Udacity)
├── README.md
├── churn_library.py (My module to modularize the process) 
├── churn_notebook.ipynb (notebook provided by Udacity, containing the process)
├── churn_script_logging_and_tests.py (My module to test the functions and log results) 
└── requirements_py3.8.txt
```

## Running Files

### How to clone the project

to clone this project, make sure you have git installed in your computer. If you have already installed git, run this command

```
git clone https://github.com/mohrosidi/udacity_customer_churn.git
```

### Dependencies

Here is a list of libraries used in this repository:

```
autopep8==1.5.7
joblib==0.11
matplotlib==2.1.0
numpy==1.12.1
pandas==0.23.3
pylint==2.9.6
scikit-learn==0.22
seaborn==0.8.1
```

To be able to run this project, you must install python library using the following command:

```
pip install -r requirements.txt
```

### Modeling

To run the workflow, simply run the `churn_library.py` in your terminal using command bellow:

```
ipython churn_library.py
```

### Testing and Logging

In other conditions, suppose you want to change the configuration of the modeling workflow, such as: changing the path of the data location, adding other models, adding feature engineering stages. You can change it in `churn_library.py` files. To test if your changes are going well, you need to do testing and logging.

To do testing and logging, you need to change a number of configurations in the `churn_script_logging_and_tests.py` file, such as: target column name, categorical column name list, data location, etc. After that, run the following command in the terminal to perform testing and loggingAfter that, run the following command in the terminal to perform testing and logging:

```
ipython churn_script_logging_and_tests.py
```

### Cleaning up your code

Make sure the code you create complies with `PEP 8` rules. To check it automatically, run pylint on the terminal.

```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

`Pylint` will provide recommendations for improvements in your code. A good code is a code that has a score close to 10.

To make repairs automatically, you can use autopep8.

```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```
