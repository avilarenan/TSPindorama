import logging
import logging.handlers

LOGGER_NAME = "TSPINDORAMA"

def get_logger():
    return logging.getLogger(LOGGER_NAME)

def setup_file_logging(log_file_path="application.log", level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):
    """
    Configures a comprehensive file-based logging system with rotation.

    Args:
        log_file_path (str): The path to the log file.
        level (int): The minimum logging level to capture (e.g., logging.DEBUG, logging.INFO).
        max_bytes (int): The maximum size of the log file in bytes before rotation.
        backup_count (int): The number of old log files to keep after rotation.
    """
    # Create a logger instance
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # Prevent duplicate log entries if called multiple times
    if not logger.handlers:
        # Create a RotatingFileHandler for log file rotation
        # This prevents the log file from growing indefinitely
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8' # Ensure proper character encoding
        )

        # Define a detailed log message format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(filename)s:%(lineno)d - %(message)s'
        )

        # Set the formatter for the file handler
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

    return logger