import logging

RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[1;41m" # Bold + Red Background
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{RESET}"
        record.msg = f"{color}{record.msg}{RESET}"
        return super().format(record)

def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s | %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

