PREREQUISITES:

Make sure Java 11 is installed and JAVA_HOME environment variable has been setup. Then install the CN protect module using the following pip command on the terminal:

      pip install cn-protect==0.9.4

The program also relies on machine learning models imported from sklearn and functionalities from numpy and pandas. So please make sure that all these python modules are also installed.

If running on a Windows machine, install the following module as well:
      pip install pyjnius

BEFORE EXECUTING THE CODE:

Make sure the directory contains the python file and the adult.csv dataset.

        -adult.csv
        -project.py

Execute the python file as follows:

        python3 project.py


Executing this creates hierarchy files which are necessary files for the anonymization algorithm.


EXPLANATION OF THE CODE:
The code is organized into modules and comments throughout the code explain how it works. The main parts of the code are as follows:


      - ML BLOCK: Contains the code for all the four ML models used and also the anonymization algorithm which processes the dataset.
      - DATA PREPROCESSING BLOCK: Contains many functions to create hierarchies which are required in the anonymization algorithm
      - PLOT function : Plots the required graphs
      - MAIN function: Makes function calls to all required models and code termination.



The code, dataset and report are available on GitHub at: https://github.com/abhishek-v/CS_588
