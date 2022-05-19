# Adapted from https://github.com/dscottboggs/python-delegate

def delegate(*args, **named_args):
    _dest = named_args.get('to')
    if _dest is None:
        raise ValueError(
            "the named argument 'to' is required on the delegate function")

    _prefix = named_args.get('prefix', '')

    def wraps(cls, *wrapped_args, **wrapped_opts):
        """Wrap the target class up in something that modifies."""
        class Wrapped(cls):

            _delegates = [(a, _dest, _prefix) for a in args] + getattr(cls, '_delegates', [])

            def __getattr__(self, name):
                """
                Return the selected name from the destination if the name is one
                of those selected. Since this is only called when `name` is not
                found in `self.__dict__`, this method should always throw an
                error when `name` is not one of the selected args to be
                delegated.
                """
                for attr, dest, prefix in self._delegates:
                    if name == prefix + attr:
                        return getattr(self.__dict__[dest], name[len(prefix):])

                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

            def __setattr__(self, name, value):
                """
                If this name is one of those selected, set it on the destination
                property. Otherwise, set it on self.
                """
                for attr, dest, prefix in self._delegates:
                    if name == prefix + attr:
                        setattr(getattr(self, dest), name[len(prefix):], value)
                        return
                self.__dict__[name] = value

            def __delattr__(self, name):
                """Delete name from `dest` or `self`"""
                for attr, dest, prefix in self._delegates:
                    if name == prefix + attr:
                        delattr(getattr(self, dest), name[len(prefix):])
                        return

                del self.__dict__[name]

        Wrapped.__doc__ = cls.__doc__ or \
            f"{cls.__class__} wrapped to delegate {args} to its {_dest} property"
        Wrapped.__repr__ = cls.__repr__
        Wrapped.__str__ = cls.__str__
        Wrapped.__name__ = cls.__name__
        return Wrapped
    return wraps
