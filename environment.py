from __future__ import annotations
from typing import Optional, Dict

from scheme import SSym, SExp, SNum, SPair, Nil


class GlobalEnvironment:
    """Singleton class for accessing the global environment.

    >>> GlobalEnvironment.get()[SSym('spam')] = SNum(42)
    >>> SSym('spam') in GlobalEnvironment.get()
    True
    >>> GlobalEnvironment.get()[SSym('spam')]
    SNum(value=42)
    """
    @staticmethod
    def get() -> Environment:
        if GlobalEnvironment._instance is None:
            GlobalEnvironment._instance = Environment(None)

        return GlobalEnvironment._instance

    _instance: Optional[Environment] = None


class Environment:
    """Return a new environment consisting of a single empty frame.

    Doctests adopted from EECS 490 Scheme Interpreter project.

    >>> env = Environment(None)
    >>> env[SSym('x')] = SNum(3)
    >>> env[SSym('bad_key')]
    Traceback (most recent call last):
        ...
    Exception: unknown identifier bad_key
    >>> env[SSym('lst')] = SPair(SNum(4), Nil)
    >>> subenv1 = env.extend()
    >>> subenv1[SSym('x')]
    SNum(value=3)
    >>> env[SSym('lst')] is subenv1[SSym('lst')]
    True
    >>> subenv1[SSym('lst')] = SPair(SNum(4), Nil)
    >>> env[SSym('lst')] is subenv1[SSym('lst')]
    False
    >>> subenv1[SSym('y')] = SNum(5)
    >>> SSym('y') in env, SSym('y') in subenv1
    (False, True)
    >>> subenv2 = env.extend()
    >>> SSym('y') in env, SSym('y') in subenv1, SSym('y') in subenv2
    (False, True, False)
    >>> env[SSym('lst')] is subenv2[SSym('lst')]
    True
    >>> subenv2[SSym('y')] = SNum(6)
    >>> subenv1[SSym('y')], subenv2[SSym('y')]
    (SNum(value=5), SNum(value=6))
    >>> subenv3 = subenv2.extend()
    >>> subenv3[SSym('x')], subenv3[SSym('y')]
    (SNum(value=3), SNum(value=6))
    >>> subenv3[SSym('z')] = SNum(7)
    >>> SSym('z') in env, SSym('z') in subenv1
    (False, False)
    >>> SSym('z') in subenv2, SSym('z') in subenv3
    (False, True)
    >>> subenv3[SSym('y')] = SNum(8)
    >>> subenv1[SSym('y')], subenv2[SSym('y')], subenv3[SSym('y')]
    (SNum(value=5), SNum(value=6), SNum(value=8))
    >>> subenv3[SSym('y')] = SNum(9)
    >>> subenv1[SSym('y')], subenv2[SSym('y')], subenv3[SSym('y')]
    (SNum(value=5), SNum(value=6), SNum(value=9))
    """
    def __init__(self, parent: Optional[Environment]):
        self._parent = parent
        self._frame: Dict[SSym, SExp] = {}

    def __getitem__(self, name: SSym):
        if name in self._frame:
            return self._frame[name]

        if self._parent is None:
            raise Exception(f'unknown identifier {name}')

        return self._parent[name]

    def __setitem__(self, name, value):
        self._frame[name] = value

    def __contains__(self, name: SSym):
        if name in self._frame:
            return True

        return self._parent is not None and name in self._parent

    def extend(self):
        return Environment(self)
