# File: mainCT4_Option1.py
# Written by: Angel Hernandez
# Description: Module 4 - Critical Thinking - Option 1
# Requirement(s): Toxicology Testing
#                 For this assignment, you will use a chemical dataset to train a neural network to predict human
#                 reaction to exposure to certain compounds. Toxicologists are very interested in the task of using
#                 machine learning to predict whether a given compound will be toxic. This task is extremely
#                 complicated because science has only a limited understanding of the metabolic processes that
#                 happen in a human body. Biologists and chemists, however, have worked out a limited set of
#                 experiments that provide indications of toxicity. If a compound is a “hit” in one of these
#                 experiments, it will likely be toxic for humans to ingest.
#                 The selected dataset is Tox21.

import os
import sys
import subprocess
import webbrowser
import time

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class HumanReactionPredictor:
    def __init__(self):
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        import numpy as np
        import deepchem as dc
        import tensorflow as tf
        from sklearn.metrics import accuracy_score as ac
        from tensorflow.keras import layers as l, models as m
        self.__np = np
        self.__tf = tf
        self.__dc = dc
        self.__layers = l
        self.__models = m
        self.test_X = None
        self.test_y = None
        self.test_w = None
        self.train_X = None
        self.train_y = None
        self.train_w = None
        self.valid_X = None
        self.valid_y = None
        self.valid_w = None
        self.__accuracy_score = ac
        self.__log_dir = "logs/tox21_tf2"
        self.__load_dataset()

    def __load_dataset(self):
        self.__np.random.seed(456)
        self.__tf.random.set_seed(456)
        _, (train, valid, test), transformers = self.__dc.molnet.load_tox21()
        self.train_X, self.train_y, self.train_w = train.X, train.y, train.w
        self.valid_X, self.valid_y, self.valid_w = valid.X, valid.y, valid.w
        self.test_X, self.test_y, self.test_w = test.X, test.y, test.w
        # Keep only the first task
        self.train_y = self.train_y[:, 0]
        self.valid_y = self.valid_y[:, 0]
        self.test_y = self.test_y[:, 0]

    # Steps 3-7 (as per instructions)
    def __build_model_with_dropout(self):
        s = self.train_X.shape[1]

        self.model = self.__models.Sequential([
            self.__layers.Input(shape=(s,)),
            self.__layers.Dense(50, activation="relu"),
            self.__layers.Dropout(0.5),
            self.__layers.Dense(1, activation="sigmoid")
        ])

        self.model.compile(optimizer=self.__tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss="binary_crossentropy",
                           metrics=["accuracy"])

    def __enable_logging_and_train_model(self):
        tb_callback = self.__tf.keras.callbacks.TensorBoard(log_dir=self.__log_dir)
        self.history = self.model.fit(self.train_X, self.train_y,
                                      validation_data=(self.valid_X, self.valid_y),
                                      epochs=10, batch_size=100, callbacks=[tb_callback], verbose=1)

    def __evaluate_accuracy(self):
        valid_pred = (self.model.predict(self.valid_X) > 0.5).astype(int).flatten()
        valid_acc = self.__accuracy_score(self.valid_y, valid_pred)
        print("Validation Accuracy:", valid_acc)
        test_pred = (self.model.predict(self.test_X) > 0.5).astype(int).flatten()
        test_acc = self.__accuracy_score(self.test_y, test_pred)
        print("Test Accuracy:", test_acc)

    def __launch_tensorboard(self):
        logdir = os.path.abspath(self.__log_dir)
        # Determine platform
        is_windows = os.name == "nt"
        # Build the correct TensorBoard command
        if is_windows:
            # Windows: tensorboard.exe lives in Python310/Scripts/
            tb_exe = os.path.join(
                os.path.dirname(sys.executable),
                "Scripts",
                "tensorboard.exe"
            )
            cmd = [tb_exe, "--logdir", logdir, "--port", "6006"]
        else:
            # Linux/macOS: tensorboard is usually in /home/local/USER/bin
            cmd = [os.path.expanduser("~/.local/bin/tensorboard"), "--logdir", logdir, "--port", "6006"]
        print("\nLaunching TensorBoard with:", " ".join(cmd))

        try:
            # Start TensorBoard as a normal external process
            process = subprocess.Popen(cmd)
            # Give TensorBoard time to start
            time.sleep(3)
            webbrowser.open("http://localhost:6006")
            print("\nTensorBoard is running at http://localhost:6006")
            print("Press ENTER to stop TensorBoard and continue...")
            input()  # Wait for user input
            print("Stopping TensorBoard...")
            process.terminate()
            print("TensorBoard stopped.\n")
        except FileNotFoundError:
            print("\nERROR: TensorBoard executable not found.")
            print("Try installing it with:")
            print("  pip install tensorboard\n")

    @staticmethod
    def suppress_warnings():
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def predict(self):
        self.__build_model_with_dropout()
        self.__enable_logging_and_train_model()
        self.__evaluate_accuracy()
        self.__launch_tensorboard()

class TestCaseRunner:
    @staticmethod
    def run_test():
        human_reaction_predictor = HumanReactionPredictor()
        human_reaction_predictor.predict()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        HumanReactionPredictor.suppress_warnings()
        dependencies = ['numpy', 'pandas', 'tensorflow', 'scikit-learn', 'tensorboard', 'deepchem']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 4 - Critical Thinking - Option 1 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()