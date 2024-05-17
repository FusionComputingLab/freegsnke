import freegsfast


class Machine(freegsfast.machine.Machine):
    """Same as freegsfast.machine.Machine.
    It can have an additional freegsfast.machine.Wall object which specifies the limiter's properties.
    """

    def __init__(self, coils, wall=None, limiter=None):
        """Instantiates the Machine, same as freegsfast.machine.Machine.

        Parameters
        ----------
        coils : FreeGSFast coils[(label, Coil|Circuit|Solenoid]
            List of coils
        wall : FreeGSFast machine.Wall object
            It is only used to display the wall in plots.
        limiter : FreeGSFast machine.Wall object
            This is the limiter. Used to define limiter plasma configurations.
        """
        super().__init__(coils, wall)
        self.limiter = limiter
