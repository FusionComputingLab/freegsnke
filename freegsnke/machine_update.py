import freegs


class Machine(freegs.machine.Machine):
    """Same as freegs.machine.Machine. 
    It can have an additional freegs.machine.Wall object which specifies the limiter's properties.
    """
    def __init__(self, coils, wall=None, limiter=None):
        """Instantiates the Machine, same as freegs.machine.Machine.

        Parameters
        ----------
        coils : freeGS coils[(label, Coil|Circuit|Solenoid]
            List of coils
        wall : freeGS machine.Wall object
            It is only used to display the wall in plots.
        limiter : freeGS machine.Wall object
            This is the limiter. Used to define limiter plasma configurations.
        """
        super().__init__(coils, wall)
        self.limiter = limiter