# File: mainCT2_Option2.py
# Written by: Angel Hernandez
# Description: Module 2 - Critical Thinking - Option 2
# Requirement(s): Predicting Future Sales
#                 You’ll use Keras to train the neural network that will try to predict the total earnings of a new
#                 game based on these characteristics. Along with the sales_data_training.csv file, there is also a
#                 second data file called sales_data_test.csv. Links to an external site.This file is exactly like
#                 the first one. The machine learning system should only use the training dataset during the
#                 training phase. Then, you'll use the test data to check how well the neural network is working.
#                 To use this data to train a neural network, you first have to scale this data so that each value is
#                 between zero and one. Neural networks train best when data in each column is all scaled to
#                 the same range.

import os
import sys
import random
import subprocess
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class ArgumentDefinition(Generic[T]):
    def __init__(self, name: str, default_value: T):
        self.name = name
        self.value: Optional[T] = None
        self.default_value = default_value
        self.caster = type(default_value)

    def read(self):
        value = input(f"{self.name} (default {self.default_value}): ")
        if value.strip() == "":
            self.value = self.default_value
        else:
            try:
                self.value = self.caster(value)
            except Exception:
                raise ValueError(f"Invalid value for {self.name}: {value}")

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class PredictorConfig:
    def __init__(self):
       self.epochs = 0
       self.random_seed = 0
       self.is_deterministic = False
       self.test_csv = "sales_data_test.csv"
       self.training_csv = "sales_data_training.csv"
       self.proposed_csv = "proposed_new_product.csv"

       arguments = [("random_seed", ArgumentDefinition("Random Seed", 42)),
                   ("is_deterministic", ArgumentDefinition("Is Deterministic?", False)),
                   ("epochs", ArgumentDefinition("Epochs", 50)),
                   ("training_csv", ArgumentDefinition("Training CSV File", self.training_csv)),
                   ("test_csv", ArgumentDefinition("Test CSV File", self.test_csv)),
                   ("proposed_csv", ArgumentDefinition("Proposed New Product CSV File", self.proposed_csv))]

       for attr_name, arg in arguments:
           arg.read()
           setattr(self, attr_name, arg.value)

class FutureSalesPredictor:
    def __init__(self, config = None):
      if config is None:
          config = PredictorConfig()
      self.__config = config
      if self.__config.is_deterministic:
          os.environ['TF_DETERMINISTIC_OPS'] = '1'
      import numpy as np
      import tensorflow as tf
      import pandas as pd
      from sklearn.preprocessing import MinMaxScaler as scaler
      self.__np = np
      self.__tf = tf
      self.__pd = pd
      self.__scaler = None
      self.__test_data_df = None
      self.__scaled_testing = None
      self.__scaled_training = None
      self.__training_data_df = None
      self.__min_max_scaler = scaler
      self.__scaled_testing_df = None
      self.__scaled_training_df = None

      self.__initialize_randomizer()

    def __initialize_randomizer(self):
        random.seed(self.__config.random_seed)
        self.__np.random.seed(self.__config.random_seed)
        self.__tf.random.set_seed(self.__config.random_seed)

    @staticmethod
    def suppress_warnings():
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def __scale_data(self):
        print("\nScaling data ...")


    def run_pipeline(self):
        self.__scale_data()


class TestCaseRunner:
    @staticmethod
    def run_test():
        sales_predictor = FutureSalesPredictor()
        sales_predictor.run_pipeline()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        FutureSalesPredictor.suppress_warnings()
        dependencies = ['numpy', 'tensorflow', 'pandas', 'scikit-learn']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 2 - Critical Thinking - Option 2 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()