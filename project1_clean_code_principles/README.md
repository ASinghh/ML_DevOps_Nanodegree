# Predict Customer Churn

Project **Predict Customer Churn** from **Clean Coding Principles** module of Udacity's ML DevOps Engineer Nanodegree course.

## Project Description

The Churn_notebook was provided by Udacity as is in the repo. The first part of the project was to define a module (churn_library.py) to modularize the methods used in the notebook, into functions. The module is expected to follow the clean coding principles that were tought in the course.
The second part of the project was to write unit tests for the functions defined in churn_library.py (churn_script_logging_and_tests.py) and log the test results.
Detailed requirements for the project are given in this [rubric](https://review.udacity.com/#!/rubrics/3094/view).

## Project Directory Structure

The structure of this project directory tree is displayed as follows:

```
├── data
│   └── bank_data.csv (Data file)
├── dataset_characteristics (Features of the dataframe)
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
└── requirements_py3.6.txt
```

## Run the program locally

to the run the program on your local machine or environment, follow the following steps,

### Setting up environment

install all the dependencies mentioned in requirements_py3.6.txt by running the following command,

```
python -m pip install -r requirements_py3.6.txt
```

### clone the project

clone this project to your local command by running the command,

```
https://github.com/ASinghh/ML_DevOps_Nanodegree.git
```

### Change Directory

Change directory to the project directory by running the following command,

```
cd project1_clean_code_principles
```

### Modeling

To perform the actions as done in churn_notebook (data import, EDA, model training, and estimation), run the following command,

```
ipython churn_library.py
```
If you would like to use individual functions, please import them individually from churn_library.py. Play arround!

### Testing and Logging

To perform the unit tests for the functions defined in churn_library.py, and log the results, please run the following command,
```
ipython churn_script_logging_and_tests.py
```

If you would like to use pytest, please run
```
pytest churn_script_logging_and_tests.py
```
Note that the later command would preform tests and display the results on your command line, but would not log the results.

### Checking and enforcing Pep8 standards

To test compliance with Pep8 standards run the following command to get a score,

```
pylint churn_script_logging_and_tests.py
pylint churn_library.py
```

You can increase the pylint score by remedying the pointers highlighted in the output provided by the above commands.

You can also use autopep8 to remedy some fo the pointers provided by pylint, by running the following commands,

```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```

### Thank you
Thank you for going through my project. You can provide feedback or reachout to me at my [email](ashutoshsinghdce@gmail.com).
