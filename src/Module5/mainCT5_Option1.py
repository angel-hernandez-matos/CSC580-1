# File: mainCT5_Option1.py
# Written by: Angel Hernandez
# Description: Module 5 - Critical Thinking - Option 1
# Requirement(s): Improving the Accuracy of a Neural Network
#                 In this assignment, you will improve the accuracy of the deep learning model of the Tox21 model
#                 from Critical Thinking Assignment, Module 4, Option 1.

import os
import sys
import subprocess
import webbrowser
import time
import itertools
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class ArgumentDefinition(Generic[T]):
    def __init__(self, name: str, default_value: T, read_arg: bool = True):
        self.name = name
        self.read_arg = read_arg
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

class PredictorConfig:
    def __init__(self, hidden = 50, layers = 1, batch_size = 100, dropout_prob = 0.5,
                 learning_rate = 0.001, epochs = 10, seed = 456):
        self.seed = seed
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate

        arguments = [("hidden", ArgumentDefinition("Hidden", self.hidden,False)),
                     ("layers", ArgumentDefinition("Layers", self.layers,False)),
                     ("seed", ArgumentDefinition("Seed", self.seed,False)),
                     ("epochs", ArgumentDefinition("Epochs", self.epochs,False)),
                     ("batch_size", ArgumentDefinition("Batch Size", self.batch_size,False)),
                     ("dropout_prob", ArgumentDefinition("Dropout Probability", self.dropout_prob,False)),
                     ("learning_rate", ArgumentDefinition("Learning Rate", self.learning_rate,False))]

        for attr_name, arg in arguments:
            if arg.read_arg: # We don't need to read from console since we'll pass hyperparameters on instantiation
                 arg.read()
            value = arg.value if arg.value is not None else arg.default_value
            setattr(self, attr_name, value )

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
    def __init__(self, hyper_param = None):
        if hyper_param is None:
            hyper_param = PredictorConfig()
        self.hyperparameters = hyper_param

        import numpy as np
        import deepchem as dc
        import tensorflow as tf
        from sklearn.metrics import accuracy_score as ac
        from tensorflow.keras import layers as l, models as m
        self.__np = np
        self.__tf = tf
        self.__dc = dc
        self.model = None
        self.__layers = l
        self.__models = m
        self.history = None
        self.__accuracy_score = ac

        # Logging directory
        self.__log_dir = (f"logs/tox21_tf2/h{hyper_param.hidden}_L{hyper_param.layers}_lr{hyper_param.learning_rate}_do"
                          f"{hyper_param.dropout_prob}_bs{hyper_param.batch_size}_seed{hyper_param.seed}")

        self.train_X = self.train_y = self.train_w = None
        self.valid_X = self.valid_y = self.valid_w = None
        self.test_X = self.test_y = self.test_w = None
        self.__load_dataset()

    @staticmethod
    def suppress_warnings():
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def __load_dataset(self):
        self.__np.random.seed(self.hyperparameters.seed)
        self.__tf.random.set_seed(self.hyperparameters.seed)
        _, (train, valid, test), _ = self.__dc.molnet.load_tox21()
        self.train_X, self.train_y, self.train_w = train.X, train.y[:, 0], train.w[:, 0]
        self.valid_X, self.valid_y, self.valid_w = valid.X, valid.y[:, 0], valid.w[:, 0]
        self.test_X, self.test_y, self.test_w = test.X, test.y[:, 0], test.w[:, 0]

    def build_model_with_dropout(self):
        tf = self.__tf
        l = self.__layers
        m = self.__models
        input_dim = self.train_X.shape[1]
        inputs = l.Input(shape=(input_dim,))
        x = inputs

        for _ in range(self.hyperparameters.layers):
            x = l.Dense(self.hyperparameters.hidden, activation="relu")(x)
            x = l.Dropout(self.hyperparameters.dropout_prob)(x)

        outputs = l.Dense(1, activation="sigmoid")(x)
        self.model = m.Model(inputs=inputs, outputs=outputs)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters.learning_rate),
                           loss="binary_crossentropy", metrics=["accuracy"])

    def enable_logging_and_train_model(self):
        tf = self.__tf
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.__log_dir, histogram_freq=1)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=3, min_lr=1e-6)

        self.history = self.model.fit(self.train_X, self.train_y, validation_data=(self.valid_X, self.valid_y),
            epochs=self.hyperparameters.epochs, batch_size=self.hyperparameters.batch_size,
            callbacks=[tb_callback, early_stop, reduce_lr], verbose=0)

    def evaluate_only(self):
        valid_pred = (self.model.predict(self.valid_X) > 0.5).astype(int).flatten()
        test_pred = (self.model.predict(self.test_X) > 0.5).astype(int).flatten()
        valid_acc = self.__accuracy_score(self.valid_y, valid_pred)
        test_acc = self.__accuracy_score(self.test_y, test_pred)
        return valid_acc, test_acc

    def __evaluate_accuracy(self):
        val_acc, test_acc = self.evaluate_only()
        print("Validation Accuracy:", val_acc)
        print("Test Accuracy:", test_acc)

    #  Original Implementation
    def predict(self):
        self.build_model_with_dropout()
        self.enable_logging_and_train_model()
        self.__evaluate_accuracy()
        self.__launch_tensorboard()

    def full_run(self):
        # Used for best hyperparameter model
        self.predict()

    def __launch_tensorboard(self):
        logdir = os.path.abspath(self.__log_dir)
        is_windows = os.name == "nt"

        if is_windows:
            tb_exe = os.path.join(os.path.dirname(sys.executable), "Scripts", "tensorboard.exe")
            cmd = [tb_exe, "--logdir", logdir, "--port", "6006"]
        else:
            cmd = [os.path.expanduser("~/.local/bin/tensorboard"), "--logdir", logdir, "--port", "6006"]

        print("\nLaunching TensorBoard with:", " ".join(cmd))

        try:
            process = subprocess.Popen(cmd)
            time.sleep(3)
            webbrowser.open("http://localhost:6006")
            print("\nTensorBoard is running at http://localhost:6006")
            print("Press ENTER to stop TensorBoard and continue...")
            input()
            process.terminate()
            print("TensorBoard stopped.\n")
        except FileNotFoundError:
            print("\nERROR: TensorBoard executable not found.")
            print("Try installing it with: pip install tensorboard\n")

class TestCaseRunner:
    @staticmethod
    def run_test():
        # Original assignment behavior
        predictor = HumanReactionPredictor()
        predictor.predict()

    @staticmethod
    def run_best_model():
        # Let's run hyperparameter search then best model
        h, l, lr, do, bs = TestCaseRunner.__run_hyperparam_search()
        hyper_param = PredictorConfig(h, l, bs, do, lr, 40)
        predictor = HumanReactionPredictor(hyper_param)
        predictor.full_run()

    @staticmethod
    def __run_hyperparam_search():
        n_epochs = 40
        best_val = -1
        best_config = None
        n_hidden_list = [64, 128]
        n_layers_list = [1, 2]
        learning_rates = [1e-3]
        dropout_probs = [0.3, 0.5]
        batch_sizes = [64]
        seeds = [456]
        print("\nBegin HyperParameter Search...\n")

        for h, l, lr, do, bs in itertools.product(n_hidden_list, n_layers_list,
                                                  learning_rates, dropout_probs, batch_sizes):
            val_scores = []

            for seed in seeds:
                hyper_param = PredictorConfig(h, l, bs, do, lr, n_epochs)
                predictor = HumanReactionPredictor(hyper_param)
                predictor.build_model_with_dropout()
                predictor.enable_logging_and_train_model()
                val_acc, _ = predictor.evaluate_only()
                val_scores.append(val_acc)

            avg_val = float(sum(val_scores) / len(val_scores))
            print(f"Config h={h}, L={l}, lr={lr}, do={do}, bs={bs} -> avg val acc={avg_val:.4f}\n")

            if avg_val > best_val:
                best_val = avg_val
                best_config = (h, l, lr, do, bs)

        print("\nBest Configuration Found...")
        print(best_config, "with avg validation accuracy =", best_val)
        print()
        return best_config

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        HumanReactionPredictor.suppress_warnings()
        dependencies = ['numpy', 'pandas', 'tensorflow', 'scikit-learn', 'tensorboard', 'deepchem']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 5 - Critical Thinking - Option 1 ***\n')
        TestCaseRunner.run_best_model()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()