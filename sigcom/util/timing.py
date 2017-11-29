def wrapper(fn, *args, **kwargs):
    def wrapped():
        return fn(*args, **kwargs)
    return wrapped
