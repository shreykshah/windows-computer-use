import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel('TRACE')
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def setup_logging(log_file=None):
    """
    Enhanced logging setup with optional file logging and detailed tracing
    
    Args:
        log_file: Optional path to log file. If provided, all logs will be written there
                 in addition to console output
    """
    # Try to add RESULT level, but ignore if it already exists
    try:
        addLoggingLevel('RESULT', 35)  # This allows ERROR, FATAL and CRITICAL
        addLoggingLevel('TRACE', 5)   # More detailed than DEBUG
    except AttributeError:
        pass  # Level already exists, which is fine

    # Get log level from environment or use default
    log_type = os.getenv('COMPUTERUSE_LOGGING_LEVEL', 'info').lower()
    
    # Get log file from environment if not provided
    if log_file is None:
        log_file = os.getenv('COMPUTERUSE_LOG_FILE', None)

    # Clear existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    class ComputerUseFormatter(logging.Formatter):
        def format(self, record):
            if type(record.name) == str and record.name.startswith('computeruse.'):
                # Keep the module name for better debugging
                module_parts = record.name.split('.')
                if len(module_parts) > 2:
                    record.name = '.'.join(module_parts[-2:])
            return super().format(record)

    # Setup console handler
    console = logging.StreamHandler(sys.stdout)

    # Define formatters
    basic_formatter = ComputerUseFormatter('%(levelname)-8s [%(name)s] %(message)s')
    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s [%(name)s:%(filename)s:%(lineno)d] %(message)s')
    result_formatter = ComputerUseFormatter('%(message)s')

    # Configure console formatter
    if log_type == 'result':
        console.setLevel('RESULT')
        console.setFormatter(result_formatter)
    elif log_type == 'debug' or log_type == 'trace':
        console.setFormatter(detailed_formatter)
    else:
        console.setFormatter(basic_formatter)

    # Add console handler to root logger
    root.addHandler(console)

    # Setup file handler if requested
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Always use detailed formatting for file logs
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(detailed_formatter)
            
            # Set file handler to TRACE level to capture everything
            file_handler.setLevel(5)  # TRACE level
            root.addHandler(file_handler)
            print(f"Logging to file: {log_file}")
        except Exception as e:
            print(f"Error setting up file logging: {e}")

    # Set root logger level
    if log_type == 'result':
        root.setLevel('RESULT')
    elif log_type == 'trace':
        root.setLevel(5)  # TRACE level
    elif log_type == 'debug':
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)

    # Configure modules with different log levels
    modules = {
        'computeruse': root.level,
        'computeruse.agent': root.level,
        'computeruse.agent.message_manager': root.level,
        'computeruse.dom': root.level,
        'computeruse.uia': root.level,
        'computeruse.llm': root.level,
        'computeruse.controller': root.level,
    }
    
    # Create and configure loggers for each module
    for module_name, level in modules.items():
        module_logger = logging.getLogger(module_name)
        module_logger.propagate = False  # Don't propagate to prevent duplicate logs
        module_logger.addHandler(console)
        if log_file:
            module_logger.addHandler(file_handler)
        module_logger.setLevel(level)

    # Set up third-party logger levels
    third_party_loggers = {
        'httpx': logging.WARNING,
        'urllib3': logging.WARNING,
        'asyncio': logging.WARNING,
        'langchain': logging.WARNING,
        'openai': logging.INFO,      # Keep OpenAI at INFO for important messages
        'httpcore': logging.WARNING,
        'charset_normalizer': logging.ERROR,
        'anthropic._base_client': logging.WARNING,
        'PIL.PngImagePlugin': logging.ERROR,
        'pywinauto': logging.WARNING,
        'comtypes': logging.WARNING,
        'azure': logging.INFO,       # Keep Azure at INFO for connection issues
    }
    
    for logger_name, level in third_party_loggers.items():
        third_party = logging.getLogger(logger_name)
        third_party.setLevel(level)
        # Don't set propagate to False for all - allow important ones to propagate
        if level >= logging.ERROR:
            third_party.propagate = False

    # Log startup message
    logger = logging.getLogger('computeruse')
    logger.info(f'ComputerUse logging setup complete with level {log_type}')
    
    if log_file:
        logger.info(f'Detailed logs will be written to {log_file}')
    
    # Enable exception context information
    sys.excepthook = log_uncaught_exception
    
    return root


def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions to provide better error information."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupts
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    logger = logging.getLogger('computeruse')
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    

def get_debug_log_filename():
    """Generate a timestamped log filename."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"computeruse_debug_{timestamp}.log"