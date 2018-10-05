import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle


class SerializationMixin:
    """
    Mixin class the adds a `.to_file(filepath)` method and a `.from_file(filepath)` class method.

    Usage
    -----
    >>> class MyClass(SerializationMixin):
    ...     def __init__(self, x):
    ...         self.x = x
    >>>
    >>> obj = MyClass(13)
    >>> obj.to_file("./my_object.dat")
    >>> del obj
    >>>
    >>> obj_new = MyClass.from_file("./my_object.dat")
    >>> print(obj_new.x)
    13

    """
    def to_file(self, filepath):
        with gzip.open(filepath, 'w') as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, filepath):
        with gzip.open(filepath) as f:
            self = pickle.load(f)
            if not isinstance(self, cls):
                raise pickle.PicklingError("loaded object must be an instance of {}, instead received a {}".format(cls, self.__class__))
        return self
