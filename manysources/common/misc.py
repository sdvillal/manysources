# coding=utf-8
"""A jumble of seemingly useful stuff."""
from contextlib import closing
import os.path as op
import os
import inspect
import gc


def giveupthefunc(upto=1):
    frame = inspect.currentframe(upto)
    code = frame.f_code
    globs = frame.f_globals
    functype = type(lambda: 0)
    funcs = []
    for func in gc.get_referrers(code):  # Slow as hell
        if type(func) is functype:
            if getattr(func, "func_code", None) is code:
                if getattr(func, "func_globals", None) is globs:
                    funcs.append(func)
                    if len(funcs) > 1:
                        return None
    return funcs[0] if funcs else None


def home():
    # What is the equivalent of user.home in python 3?
    return op.expanduser('~')


def ensure_writable_dir(path):
    """Ensures that a path is a writable directory."""
    def check_path(path):
        if not op.isdir(path):
            raise Exception('%s exists but it is not a directory' % path)
        if not os.access(path, os.W_OK):
            raise Exception('%s is a directory but it is not writable' % path)
    if op.exists(path):
        check_path(path)
    else:
        try:
            os.makedirs(path)
        except Exception:
            if op.exists(path):  # Simpler than using a file lock to work on multithreading...
                check_path(path)
            else:
                raise
    return path


def ensure_dir(path):
    return ensure_writable_dir(path)


def run_db_statement(dburi, statement):
    """Runs a single SQL statement, always opening a new connection.

    Parameters
    ----------
    dburi : string
        URL that would allow us to connect to the database.

    statement : string
        The SQL statement to run.

    Returns
    -------
    A sqlalchemy ResultProxy or None if the statement is not a query.
    """
    import sqlalchemy  # Just use something less heavyweight
    engine = sqlalchemy.create_engine(dburi)
    # Get a connection...
    with closing(engine.connect()) as conn:
        # Move out of transaction, in case we want to do things like CREATE DATABASE
        conn.execute('commit')
        # Run the thingy
        return conn.execute(statement)


def can_connect(dburi):
    """Tests whether we can connect to a database or not.

    Parameters
    ----------
    dburi : string
        URL that would allow us to connect to the database.

    Returns
    -------
    (True, None) if we can connect, otherwise (False, exception)
    """
    import sqlalchemy  # Just use something less heavyweight
    try:
        engine = sqlalchemy.create_engine(dburi)
        with closing(engine.connect()):
            return True, None
    except Exception, e:
        return False, e