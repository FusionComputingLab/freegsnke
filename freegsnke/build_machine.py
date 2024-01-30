import os
import pickle
from freegs.machine import Circuit, Wall, Solenoid
from .machine_update import Machine
from freegs.coil import Coil
from freegs.multi_coil import MultiCoil


passive_coils_path = os.environ.get('PASSIVE_COILS_PATH', None)
if passive_coils_path is None:
    raise ValueError('PASSIVE_COILS_PATH environment variable not set.')

active_coils_path = os.environ.get('ACTIVE_COILS_PATH', None)
if active_coils_path is None:
    raise ValueError('ACTIVE_COILS_PATH environment variable not set.')

wall_path = os.environ.get('WALL_PATH', None)
if wall_path is None:
    raise ValueError('WALL_PATH environment variable not set.')

limiter_path = os.environ.get('LIMITER_PATH', None)
if limiter_path is None:
    raise ValueError('LIMITER_PATH environment variable not set.')

with open(passive_coils_path, 'rb') as f:
    passive_coils = pickle.load(f)

with open(active_coils_path, 'rb') as f:
    active_coils = pickle.load(f)

with open(wall_path, 'rb') as f:
    wall = pickle.load(f)

with open(limiter_path, 'rb') as f:
    limiter = pickle.load(f)

if 'Solenoid' not in active_coils:
    raise ValueError('No Solenoid in active coils. Must be capitalised Solenoid.')


def tokamak():
    """MAST-Upgrade, using MultiCoil to represent coils with different locations
    for each strand.
    """
    # Add the solenoid
    coils = [
        (
            "Solenoid",
            Circuit(
                [
                    (
                        "Solenoid",
                        MultiCoil(
                            active_coils['Solenoid']['R'],
                            active_coils['Solenoid']['Z']),
                            float(active_coils['Solenoid']['polarity']) * \
                            float(active_coils['Solenoid']['multiplier']),
                            ),
                ]
            ),
        ),
    ]

    # Add remaining active coils
    for coil_name in active_coils:
        if not coil_name == 'Solenoid':
            coils.append(
                (
                    coil_name,
                    Circuit(
                        [
                            (
                                coil_name+ind,
                                MultiCoil(
                                    active_coils[coil_name][ind]['R'],
                                    active_coils[coil_name][ind]['Z']
                                ),
                                float(
                                    active_coils[coil_name][ind]['polarity']
                                ) * \
                                float(
                                    active_coils[coil_name][ind]['multiplier']
                                )
                            ) for ind in active_coils[coil_name]
                        ]
                    )
                )
            )

    # Add passive coils
    for i, coil in enumerate(passive_coils):
        coil_name = f"passive_{i}"
        coils.append(
            (
                (
                    coil_name,
                    Coil(
                        R=coil['R'],
                        Z=coil['Z'],
                        area=coil['dR']*coil['dZ'],
                        control=False
                    )
                )
            )
        )

    # Add walls
    r_wall = [entry["R"] for entry in wall]
    z_wall = [entry["Z"] for entry in wall]

    # Add limiter
    r_limiter = [entry["R"] for entry in limiter]
    z_limiter = [entry["Z"] for entry in limiter]

    return Machine(coils, wall=Wall(r_wall, z_wall), limiter=Wall(r_limiter, z_limiter))


if __name__ == "__main__":
    # tokamak = tokamak()
    for coil_name in active_coils:
        print([pol for pol in active_coils[coil_name]])