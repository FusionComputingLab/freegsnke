import os
import pickle
from freegs.machine import Machine, Circuit, Wall, Solenoid
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

with open(passive_coils_path, 'rb') as f:
    passive_coils = pickle.load(f)

with open(active_coils_path, 'rb') as f:
    active_coils = pickle.load(f)

with open(wall_path, 'rb') as f:
    wall = pickle.load(f)

if 'Solenoid' not in active_coils:
    raise ValueError('No solenoid in active coils.')


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
                            active_coils['Solenoid']['polarity']
                    ),
                ]
            ),
        ),
    ]
    del active_coils['Solenoid']

    # Add remaining active coils
    for coil_name in active_coils:
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
                            float(active_coils[coil_name][ind]['polarity']) * \
                            float(active_coils[coil_name][ind]['multiplier'])
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

    return Machine(coils, Wall(r_wall, z_wall))


if __name__ == "__main__":
    # tokamak = tokamak()
    for coil_name in active_coils:
        print([pol for pol in active_coils[coil_name]])