import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from lime import lime_tabular

shap.initjs()


class FraudModelExplainer:
    """
    A class to explain machine learning models using SHAP and LIME.

    Attributes:
    -----------
    model : object
        The trained machine learning model.
    X_test : DataFrame
        The test dataset for generating explanations.

    Methods:
    --------
    explain_with_shap(instance_idx=0, cmap='Blues'):
        Generates SHAP Summary Plot, Force Plot, and Dependence Plot with custom color.

    explain_with_lime(instance_idx=0):
        Generates LIME Feature Importance Plot for a single instance with custom color.

    explain_model(instance_idx=0, cmap='Blues'):
        Runs both SHAP and LIME explainability functions with optional custom color.
    """

    def __init__(self, model_path, X_test):
        """
        Initialize the explainer with the model and test dataset.

        Parameters:
        -----------
        model_path : str
            Path to the saved model file (.pkl).
        X_test : DataFrame
            The test dataset in pandas DataFrame format.
        """
        self.model = joblib.load(model_path)
        self.X_test = X_test

        # Extract the actual model if inside a pipeline
        if hasattr(self.model, "steps"):
            self.model = self.model.steps[-1][1]

        # Ensure X_test is a DataFrame
        if not isinstance(self.X_test, pd.DataFrame):
            raise ValueError("X_test must be a pandas DataFrame.")

    def _create_shap_explainer(self):
        """
        Automatically detects model type and creates a suitable SHAP explainer.
        """
        model_type = type(self.model).__name__.lower()

        if "tree" in model_type:
            return shap.TreeExplainer(self.model)
        elif "linear" in model_type:
            return shap.LinearExplainer(self.model, self.X_test)
        else:
            # KernelExplainer as a fallback for models not directly supported by SHAP
            return shap.KernelExplainer(
                self.model.predict, shap.sample(self.X_test, 100)
            )

    def explain_with_shap(self, instance_idx=0):
        """
        Generate SHAP Summary Plot, Force Plot, and Dependence Plot for the model.

        Parameters:
        -----------
        instance_idx : int, optional (default=0)
            The index of the instance to explain with SHAP Force Plot.
        """
        print("Generating SHAP explanations...")

        model = self.model
        explainer = shap.TreeExplainer(model, self.X_test)
        shap_values = explainer.shap_values(self.X_test)

        # Print type and shape for debugging
        print(f"Type of SHAP values: {type(shap_values)}")
        print(f"Shape of SHAP values: {shap_values.shape}")

        # SHAP Summary Plot: Overview of important features
        plt.figure(figsize=(15, 4))
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title("SHAP Summary Plot")
        plt.show()

        # Plot SHAP force plot for the selected instance from the test data
        # Use `shap.plots.force` correctly
        shap.plots.force(
            explainer.expected_value,
            shap_values[instance_idx],
            feature_names=self.X_test.columns,
            matplotlib=True,
        )

        # SHAP Dependence Plot: Relationship between feature and model output
        shap.dependence_plot(
            self.X_test.columns[0], shap_values, self.X_test, show=False
        )
        plt.title(f"SHAP Dependence Plot for Feature: {self.X_test.columns[0]}")
        plt.show()

    def explain_with_lime(self, instance_idx=0):
        """
        Generates LIME Feature Importance Plot with custom color.

        Parameters:
        -----------
        instance_idx : int (default=0)
            The index of the instance to explain using LIME.
        """
        print("Generating LIME explanations...")

        explainer_lime = lime_tabular.LimeTabularExplainer(
            training_data=self.X_test.values,
            feature_names=self.X_test.columns,
            mode="classification",
        )

        # Ensure instance index is within range
        if instance_idx >= self.X_test.shape[0]:
            raise IndexError(f"Instance index {instance_idx} is out of range.")

        instance = self.X_test.iloc[instance_idx].values
        explanation = explainer_lime.explain_instance(
            instance, self.model.predict_proba
        )

        # LIME Plot with custom color
        fig = explanation.as_pyplot_figure()
        for bar in fig.axes[0].patches:  
            bar.set_facecolor("teal")
        plt.title(f"LIME Feature Importance for Instance {instance_idx}")
        plt.show()

    def explain_model(self, instance_idx=0, cmap="Blues"):
        """
        Runs both SHAP and LIME explainability methods with optional custom color.

        Parameters:
        -----------
        instance_idx : int (default=0)
            The index of the instance to explain.
        cmap : str (default='Blues')
            The colormap to be used for SHAP plots.
        """
        self.explain_with_shap(instance_idx, cmap)
        self.explain_with_lime(instance_idx)
