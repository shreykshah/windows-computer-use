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


def setup_logging():
    # Try to add RESULT level, but ignore if it already exists
    try:
        addLoggingLevel('RESULT', 35)  # This allows ERROR, FATAL and CRITICAL
    except AttributeError:
        pass  # Level already exists, which is fine

    log_type = os.getenv('COMPUTERUSE_LOGGING_LEVEL', 'info').lower()

    # Check if handlers are already set up
    if logging.getLogger().hasHandlers():
        return

    # Clear existing handlers
    root = logging.getLogger()
    root.handlers = []

    class ComputerUseFormatter(logging.Formatter):
        def format(self, record):
            if type(record.name) == str and record.name.startswith('computeruse.'):
                record.name = record.name.split('.')[-2]
            return super().format(record)

    # Setup single handler for all loggers
    console = logging.StreamHandler(sys.stdout)

    # additional setLevel here to filter logs
    if log_type == 'result':
        console.setLevel('RESULT')
        console.setFormatter(ComputerUseFormatter('%(message)s'))
    else:
        console.setFormatter(ComputerUseFormatter('%(levelname)-8s [%(name)s] %(message)s'))

    # Configure root logger only
    root.addHandler(console)

    # switch cases for log_type
    if log_type == 'result':
        root.setLevel('RESULT')  # string usage to avoid syntax error
    elif log_type == 'debug':
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)

    # Configure computeruse logger
    computeruse_logger = logging.getLogger('computeruse')
    computeruse_logger.propagate = False  # Don't propagate to root logger
    computeruse_logger.addHandler(console)
    computeruse_logger.setLevel(root.level)  # Set same level as root logger

    logger = logging.getLogger('computeruse')
    logger.info('ComputerUse logging setup complete with level %s', log_type)
    # Silence third-party loggers
    for logger in [
        'httpx',
        'urllib3',
        'asyncio',
        'langchain',
        'openai',
        'httpcore',
        'charset_normalizer',
        'anthropic._base_client',
        'PIL.PngImagePlugin',
    ]:
        third_party = logging.getLogger(logger)
        third_party.setLevel(logging.ERROR)
        third_party.propagate = False