import logging.config


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s : %(name)s : %(levelname)s : %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "simple",
            "stream": "ext://sys.stderr"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": "pdf_appRecognizer_data.log",
            "encoding": "utf-8",
            "mode": "w"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": [
            "console",
            "file"
        ]
    },
}


def activate_logging():
    logging.config.dictConfig(LOGGING)
