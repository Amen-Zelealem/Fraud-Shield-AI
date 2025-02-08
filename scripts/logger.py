import logging
import os

class LoggerSetup:
    """
    A class to configure and manage logging for the application.

    Attributes:
    ----------
    log_file : str
        The file where logs will be saved.
    log_level : int
        The level of logging (default is logging.INFO).
    """

    def __init__(self, log_file='logs/app.log', log_level=logging.INFO):
        """
        Initializes the logger with a specified log file and logging level.

        Parameters:
        ----------
        log_file : str
            The path to the file where logs will be saved.
        log_level : int, optional
            The logging level (default is logging.INFO).
        """
        self.log_file = log_file
        self.log_level = log_level
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Configures the logger with the specified settings.

        Returns:
        -------
        logging.Logger
            The configured logger instance.
        """
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        os.makedirs(log_dir, exist_ok=True)

        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)

        # Create a file handler for logging
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)

        # Define the logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        # Optionally, add a console handler for real-time logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.info("Logger configured successfully.")
        
        return logger

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger.

        Returns:
        -------
        logging.Logger
            The configured logger instance.
        """
        return self.logger