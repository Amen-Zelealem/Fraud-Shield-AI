import os
import time
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.keras
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def plot_class_distribution(class_counts, ax, dataset_name):
    """Plots the class distribution with total numbers on top of each bar."""
    colors = sns.color_palette("pastel", n_colors=len(class_counts))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        hue=class_counts.index,
        palette=colors,
        ax=ax,
        legend=True,
    )
    ax.set_title(f"{dataset_name} Class Distribution")
    ax.set_ylabel("Number of Instances")
    ax.set_xlabel("Class")

    # Remove y-axis tick labels
    ax.set_yticklabels([])

    # Annotate the total number on top of each bar
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=12,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )


def visualize_class_distributions(fraud_data, credit_data):
    """Generates and displays class distribution plots for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    fraud_class_counts = fraud_data["class"].value_counts()
    creditcard_class_counts = credit_data["Class"].value_counts()

    plot_class_distribution(fraud_class_counts, axes[0], "Fraud Ecommerce Dataset")
    plot_class_distribution(
        creditcard_class_counts, axes[1], "Credit Card Bank Dataset"
    )

    plt.tight_layout()
    plt.show()


# Set MLflow tracking URI to store mlruns folder in the app directory
mlflow.set_tracking_uri("../app/mlruns")


class FraudDetectionPipeline:
    """
    Machine Learning pipeline for fraud detection, including model selection,
    training, evaluation, and logging with MLflow.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.performance_metrics = {}
        self.y_probs = {}

    def add_models(self):
        """Adds machine learning models to the pipeline."""
        self.models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "LSTM": self.build_lstm_model(),
            "CNN": self.build_cnn_model(),
        }

    def build_lstm_model(self):
        """Builds an LSTM model."""
        model = Sequential(
            [
                Input(shape=(self.X_train.shape[1], 1)),
                LSTM(50),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def build_cnn_model(self):
        """Builds a CNN model."""
        model = Sequential(
            [
                Input(shape=(self.X_train.shape[1], 1)),
                Conv1D(filters=32, kernel_size=3, activation="relu"),
                Flatten(),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def hyperparameter_tuning(self):
        """Performs hyperparameter tuning for Random Forest and Gradient Boosting models."""
        param_grids = {
            "Random Forest": {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [None, 5, 10],
            },
            "Gradient Boosting": {
                "classifier__learning_rate": [0.01, 0.1],
                "classifier__n_estimators": [50, 100],
            },
        }

        for name, model in self.models.items():
            if name in ["LSTM", "CNN"]:
                continue  # Skip tuning for neural networks

            print(f"Tuning {name}...")
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])

            search = GridSearchCV(
                pipeline,
                param_grid=param_grids[name],
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
            )
            search.fit(self.X_train, self.y_train)
            self.models[name] = search.best_estimator_
            print(f"Best parameters for {name}: {search.best_params_}")

    def train_and_evaluate(self):
        """Trains, evaluates models, and logs results with MLflow."""
        self.add_models()
        self.hyperparameter_tuning()

        best_model, best_score, best_model_name = None, 0, ""

        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                start_time = time.time()

                if name in ["LSTM", "CNN"]:
                    X_train_reshaped = self.X_train.values.reshape(
                        -1, self.X_train.shape[1], 1
                    )
                    X_test_reshaped = self.X_test.values.reshape(
                        -1, self.X_test.shape[1], 1
                    )

                    model.fit(
                        X_train_reshaped,
                        self.y_train,
                        epochs=5,
                        batch_size=32,
                        verbose=0,
                    )
                    y_prob = model.predict(X_test_reshaped).flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_prob = model.predict_proba(self.X_test)[:, 1]

                end_time = time.time()
                print(f"{name} trained in {end_time - start_time:.2f} seconds")

                metrics = {
                    "Accuracy": accuracy_score(self.y_test, y_pred),
                    "Precision": precision_score(self.y_test, y_pred),
                    "Recall": recall_score(self.y_test, y_pred),
                    "F1 Score": f1_score(self.y_test, y_pred),
                    "ROC AUC": roc_auc_score(self.y_test, y_prob),
                }
                self.performance_metrics[name] = metrics
                self.y_probs[name] = y_prob

                for metric, value in metrics.items():
                    mlflow.log_metric(metric.lower().replace(" ", "_"), value)

                if name in ["Random Forest", "Gradient Boosting"]:
                    mlflow.log_params(self.models[name].get_params())

                model_name = name.replace(" ", "_").lower()
                mlflow_model_path = f"{model_name}_model"

                if name in ["LSTM", "CNN"]:
                    mlflow.keras.log_model(model, mlflow_model_path)
                else:
                    mlflow.sklearn.log_model(model, mlflow_model_path)

                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/{mlflow_model_path}", name
                )

                if metrics["ROC AUC"] > best_score:
                    best_score, best_model, best_model_name = (
                        metrics["ROC AUC"],
                        model,
                        name,
                    )

                print(f"{name} logged in MLflow")

        return best_model, best_model_name

    def save_best_model(self, best_model, best_model_name, dataset_name):
        """Saves the best model to disk."""
        model_path = f"../app/{best_model_name.replace(' ', '_').lower()}_{dataset_name}_best_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"Best model ({best_model_name}) saved at {model_path}")

    def get_results(self):
        """Returns model evaluation results."""
        return self.performance_metrics, self.y_probs
