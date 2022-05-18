# Taken from https://github.com/dscottboggs/python-delegate

def bind(instance, func, name):
    """Turn a function into a bound method on instance"""
    setattr(instance, name, func.__get__(instance, instance.__class__))

def delegate(*args, **named_args):
    dest = named_args.get('to')
    if dest is None:
        raise ValueError(
            "the named argument 'to' is required on the delegate function")
    def wraps(cls, *wrapped_args, **wrapped_opts):
        """Wrap the target class up in something that modifies."""
        class Wrapped(cls):

            def __getattr__(self, name):
                """
                Return the selected name from the destination if the name is one
                of those selected. Since this is only called when `name` is not
                found in `self.__dict__`, this method should always throw an
                error when `name` is not one of the selected args to be
                delegated.
                """
                if name in args: return getattr(self.__dict__[dest], name)
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

            def __setattr__(self, name, value):
                """
                If this name is one of those selected, set it on the destination
                property. Otherwise, set it on self.
                """
                if name in args: setattr(getattr(self, dest), name, value)
                else: self.__dict__[name] = value

            def __delattr__(self, name):
                """Delete name from `dest` or `self`"""
                if name in args: delattr(getattr(self, dest), name)
                else: del self.__dict__[name]

            def __init__(self, *wrapped_args, **wrapped_opts):
                super().__init__(*wrapped_args, **wrapped_opts)

        Wrapped.__doc__ = cls.__doc__ or \
            f"{cls.__class__} wrapped to delegate {args} to its {dest} property"
        Wrapped.__repr__ = cls.__repr__
        Wrapped.__str__ = cls.__str__
        Wrapped.__name__ = cls.__name__
        return Wrapped
    return wraps
