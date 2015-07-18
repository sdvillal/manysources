# coding=utf-8
"""Many sources experiments"""
import os.path as op
import logging

# Current version of the package
__version__ = '1.0.0'

# Where the project resides
MANYSOURCES_ROOT = op.realpath(op.join(op.dirname(__file__), '..'))

# Common logger for the manysources code
_logger = logging.getLogger('manysources')
_logger.setLevel(logging.DEBUG)
debug = _logger.debug
info = _logger.info
warning = _logger.warning
error = _logger.error
logging.basicConfig()