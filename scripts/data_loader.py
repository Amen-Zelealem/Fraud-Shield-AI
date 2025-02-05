import pandas as pd
import logging

class DatasetLoader:
    """
    A class for loading and preprocessing a dataset, including cleaning and handling missing values.

    Attributes:
    ----------
    filepath : str
        The file path of the dataset.
    logger : logging.Logger
        The logger instance for logging actions and errors.
    data : pd.DataFrame, optional
        The dataset loaded from the file path.
    """

    def __init__(self, filepath: str, logger: logging.Logger):
        """
        Initializes the DatasetLoader with a dataset filepath and logger.

        Parameters:
        ----------
        filepath : str
            The path to the dataset file (CSV format).
        logger : logging.Logger
            A logger instance for logging information and errors.
        """
        self.filepath = filepath
        self.logger = logger
        self.data = None
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified filepath.

        Returns:
        -------
        pd.DataFrame or None
            The loaded dataset as a DataFrame, or None if an error occurs.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            self.logger.info("Dataset loaded successfully.")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return None  # Return None if there's an error loading the dataset

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans the dataset by handling missing values and duplicates.

        Returns:
        -------
        pd.DataFrame
            The cleaned dataset as a DataFrame.
        """
        if self.data is not None:
            initial_shape = self.data.shape
            self.data.dropna(inplace=True)  # Remove missing values
            self.data.drop_duplicates(inplace=True)  # Remove duplicates
            self.logger.info(f"Data cleaned: {initial_shape} -> {self.data.shape}")
            return self.data
        else:
            self.logger.warning("No data to clean.")
            return None