class Trap(Exception):
    """A trap exception thrown by the `TrapInst` instruction."""
    pass


class EnvBindingNotFound(Exception):
    """
    An exception that indicates that a requested symbol does not
    exist in an environment.
    """
    pass
