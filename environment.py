from __future__ import annotations

from typing import Dict, Generic, Optional, TypeVar

from scheme import Nil, SExp, SFunction, SNum, SPair, SSym, parse
from visitor import Visitor


class EnvAssigner(Visitor):
    """Creates, fills, and assigns environments to expressions in the program.
    >>> class EnvPrinter(Visitor):
    ...     def visit_SExp(self, expr: SExp):
    ...         print(expr)
    ...         print(expr.environment)
    ...         super().visit_SExp(expr)
    >>> code = '(define (spam egg) egg)'
    >>> parsed = parse(code)
    >>> EnvAssigner(GlobalEnvironment.get()).visit(parsed)
    >>> EnvPrinter().visit(parsed) #doctest: +ELLIPSIS
    SFunction(name=SSym(name='spam'),... is_lambda=False)
    {
        egg: Nil
    }
    spam
    {
        egg: Nil
    }
    egg
    {
        egg: Nil
    }
    egg
    {
        egg: Nil
    }

    >>> code = '(define (spam egg) ((lambda (sausage) sausage) 42))'
    >>> parsed = parse(code)
    >>> EnvAssigner(GlobalEnvironment.get()).visit(parsed)
    >>> EnvPrinter().visit(parsed) #doctest: +ELLIPSIS
    SFunction(name=SSym(name='spam'),... is_lambda=False)
    {
        egg: Nil
    }
    spam
    {
        egg: Nil
    }
    egg
    {
        egg: Nil
    }
    SCall(func=SFunction(name=SSym(name='__lambda0'),...), args=[SNum(value=42)])
    {
        egg: Nil
    }
    SFunction(name=SSym(name='__lambda0'),... is_lambda=True)
    {
        sausage: Nil
    }
    sausage
    {
        sausage: Nil
    }
    sausage
    {
        sausage: Nil
    }
    42
    {
        egg: Nil
    }

    """

    def __init__(self, parent_env: Environment):
        self._parent_env: Environment = parent_env

    def visit_SExp(self, expr: SExp) -> None:
        expr.environment = self._parent_env
        super().visit_SExp(expr)

    def visit_SFunction(self, func: SFunction) -> None:
        func_env = self._parent_env.extend()

        for param in func.params:
            func_env[param] = Nil

        func.environment = func_env

        current_parent_env = self._parent_env
        self._parent_env = func_env
        super().visit_SFunction(func)
        self._parent_env = current_parent_env


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


EnvValType = TypeVar('EnvValType')


class Environment(Generic[EnvValType]):
    """Return a new environment consisting of a single empty frame.

    Doctests adopted from EECS 490 Scheme Interpreter project.

    >>> env = Environment(None)
    >>> env[SSym('x')] = SNum(3)
    >>> env[SSym('bad_key')]
    Traceback (most recent call last):
        ...
    environment.EnvBindingNotFound: unknown identifier bad_key
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
    def __init__(self, parent: Optional[Environment[EnvValType]]):
        self._parent = parent
        self._frame: Dict[SSym, EnvValType] = {}

    def __getitem__(self, name: SSym) -> EnvValType:
        if name in self._frame:
            return self._frame[name]

        if self._parent is None:
            raise EnvBindingNotFound(f'unknown identifier {name}')

        return self._parent[name]

    def __setitem__(self, name: SSym, value: EnvValType) -> None:
        self._frame[name] = value

    def __contains__(self, name: SSym) -> bool:
        if name in self._frame:
            return True

        return self._parent is not None and name in self._parent

    def in_local(self, name: SSym) -> bool:
        """Returns true if name exists in the innermost frame.
        >>> env = Environment(None)
        >>> local_env = env.extend()
        >>> env[SSym('spam')] = Nil
        >>> local_env[SSym('egg')] = Nil
        >>> SSym('spam') in local_env
        True
        >>> local_env.in_local(SSym('spam'))
        False
        >>> local_env.in_local(SSym('egg'))
        True
        """
        return name in self._frame

    def extend(self) -> Environment[EnvValType]:
        return Environment(self)

    def __str__(self) -> str:
        if len(self._frame) == 0:
            return '{}'

        body = '\n'.join(
            (f'    {key}: {value}' for key, value in self._frame.items())
        )
        return f"{{\n{body}\n}}"


class EnvBindingNotFound(Exception):
    """
    An exception that indicates that a requested symbol does not
    exist in an Environment.
    """
    pass
