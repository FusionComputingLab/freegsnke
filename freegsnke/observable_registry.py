from typing import Any, Callable, TypeAlias

ObservableFunc: TypeAlias = Callable[[float], Any]


class ObservableRegistry:
    """
    An observable registry provides an abstracted means of obtaining observables
    regarding an equilibrium and associated parameters. The idea is that any diagnostic
    or postprocessing logic that uses this class to obtain parameters it needs does not
    need to know how those parameters were obtained. For example, `betap` may be
    obtained in a fully precise way from a FreeGNSKE equilibrium, but it may also be
    calculated from noisy diagnositcs, or predicted using an emulator.

    TODO(Matthew): I don't love how this is implemented, we need to be able to make this
                    useful generally and so it should support getting across the history
                    of an equilibrium and its evolution. As such, we maybe shouldn't
                    take timestamp in `get` but rather something that better reflects
                    other uses. This would then change the Callable definition.
    """

    def __init__(self):
        # The registry is really just a dictionary of named functions called via the
        # ObservableRegistry.get method.
        self._observables: dict[str, ObservableFunc] = {}

    def register(self, name: str, fn: ObservableFunc):
        """
        Register the named observable to be computed using the provided function.

        Parameters
        ----------
        name : str
            The name of the observable to register.
        fn : ObservableFunc
            The function with which the named observable is computed.
        """
        self._observables[name] = fn

    def has(self, name: str) -> bool:
        """
        Determines if an observable with the given name has been registered.

        Parameters
        ----------
        name : str
            The name of the observable to determine registration of.
        """
        return name in self._observables

    def get(self, name: str, timestamp: float) -> Any | None:
        """
        Obtains the value of the given observable at the given timestamp. Note that
        """
        if name not in self._observables:
            return None

        return self._observables[name](timestamp)
