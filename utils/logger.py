import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler

def setup_logging(level=logging.INFO, log_to_file=True, log_dir="logs", main_process=False):
    """
    Sets up logging for the application with RichHandler for console
    and an optional rotating file handler.
    """
    log_format = "%(asctime)s - %(processName)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"

    # Get the root logger
    root_logger = logging.getLogger()

    # Avoid adding handlers multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Set the base level for the logger
    root_logger.setLevel(level)

    # Rich Handler for console output
    console_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_path=False, # The format string already includes the path
        log_time_format="[%X]"
    )
    console_handler.setFormatter(logging.Formatter("%(message)s")) # Rich handles the rest
    root_logger.addHandler(console_handler)

    # File Handler for file output, only in the main process
    if log_to_file and main_process:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, "app.log")

        # Use a rotating file handler to prevent log files from growing indefinitely
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10 MB per file, 5 backups
        )
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    logging.info("Logging initialized.")


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


if __name__ == '__main__':
    setup_logging(level=logging.DEBUG, main_process=True)

    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")

    try:
        1 / 0
    except Exception as e:
        logging.exception("An exception occurred.")
