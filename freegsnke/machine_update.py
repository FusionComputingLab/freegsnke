"""
Defines the FreeGSNKE machine object, which inherits from the FreeGS4E machine object. 

Copyright 2024 Nicola C. Amorisco, George K. Holt, Kamran Pentland, Adriano Agnello, Alasdair Ross, Matthijs Mars.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
"""

import freegs4e


class Machine(freegs4e.machine.Machine):
    """Same as freegs4e.machine.Machine.
    It can have an additional freegs4e.machine.Wall object which specifies the limiter's properties.
    """

    def __init__(self, coils, wall=None, limiter=None):
        """Instantiates the Machine, same as freegs4e.machine.Machine.

        Parameters
        ----------
        coils : FreeGS4E coils[(label, Coil|Circuit|Solenoid]
            List of coils
        wall : FreeGS4E machine.Wall object
            It is only used to display the wall in plots.
        limiter : FreeGS4E machine.Wall object
            This is the limiter. Used to define limiter plasma configurations.
        """
        super().__init__(coils, wall)
        self.limiter = limiter
