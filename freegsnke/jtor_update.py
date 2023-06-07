import freegs
from freegs import critical


class ConstrainBetapIp(freegs.jtor.ConstrainBetapIp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'fast'):
            self.Jtor = self._Jtor

    def _Jtor(self, R, Z, psi, psi_bndry=None):
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)
        return self.jtor
    
class ConstrainPaxisIp(freegs.jtor.ConstrainPaxisIp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'fast'):
            self.Jtor = self._Jtor  

    def _Jtor(self, R, Z, psi, psi_bndry=None):
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)
        return self.jtor