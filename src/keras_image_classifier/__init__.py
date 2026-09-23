"""Reproducible image classification with Keras 3."""

import os

# Keras reads the backend once, at first import. Anything already chosen by the
# user (environment variable) wins; otherwise use the backend this project is tested on.
os.environ.setdefault("KERAS_BACKEND", "jax")

__version__ = "1.1.0"


class KicError(Exception):
    """A problem the user can fix: bad path, bad argument, unusable data.

    The command line prints the message and exits with status 2, without a traceback.
    """
