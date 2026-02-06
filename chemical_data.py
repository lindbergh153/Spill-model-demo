"""
Chemical Property Database
==========================

Load and manage thermodynamic properties of chemical compounds
from TAMOC database for oil/gas mixture calculations.

"""

from __future__ import annotations
from pathlib import Path

import pandas as pd


def tamoc_data():
    """Load TAMOC chemical property database."""
    data_path = Path(__file__).parent / 'data' / 'ChemData.csv'
    data_chem = pd.read_csv(data_path, skiprows=[2])

    composition = data_chem.iloc[:, 0][1:].tolist()
    M = [float(i) for i in data_chem.iloc[:, 1][1:].tolist()]
    Pc = [float(i) for i in data_chem.iloc[:, 2][1:].tolist()]
    Tc = [float(i) for i in data_chem.iloc[:, 3][1:].tolist()]
    Vc = [float(i) for i in data_chem.iloc[:, 4][1:].tolist()]
    Tb = [float(i) for i in data_chem.iloc[:, 5][1:].tolist()]
    Vb = [float(i) for i in data_chem.iloc[:, 6][1:].tolist()]
    omega = [float(i) for i in data_chem.iloc[:, 7][1:].tolist()]
    kh_0 = [float(i) for i in data_chem.iloc[:, 8][1:].tolist()]
    neg_dH_solR = [float(i) for i in data_chem.iloc[:, 9][1:].tolist()]
    nu_bar = [float(i) for i in data_chem.iloc[:, 10][1:].tolist()]
    B = [float(i) for i in data_chem.iloc[:, 11][1:].tolist()]
    dE = [float(i) for i in data_chem.iloc[:, 12][1:].tolist()]
    K_salt = [float(i) for i in data_chem.iloc[:, 13][1:].tolist()]

    # Create an empty dictionary of chemical property data
    data = {}
    # Fill the dictionary with the properties for each chemical component
    for i in range(len(composition)):
        # Add this chemical
        data[composition[i]] = {
            'M': M[i],
            'Pc': Pc[i],
            'Tc': Tc[i],
            'Vc': Vc[i],
            'Tb': Tb[i],
            'Vb': Vb[i],
            'omega': omega[i],
            'kh_0': kh_0[i],
            '-dH_solR': neg_dH_solR[i],
            'nu_bar': nu_bar[i],
            'B': B[i],
            'dE': dE[i],
            'K_salt': K_salt[i]
        }

    units = {}
    read_units = {}
    for i in range(len(data_chem.iloc[0, 1:-1].tolist())):
        units[data_chem.columns[i + 1]] = data_chem.iloc[0, 1:-1].tolist()[i]
        read_units[data_chem.columns[i + 1]] = data_chem.iloc[0, 1:-1].tolist()[i]

    # Convert to SI units.  If you add a new unit to the file ChemData.csv,
    # then you should include a check for it here.
    for chemical in data:
        for variable in units:
            if read_units[variable].find('g/mol') >= 0:
                # Convert to kg/mol
                data[chemical][variable] /= 1000.
                units[variable] = '(kg/mol)'

            if read_units[variable].find('psia') >= 0:
                # Convert to Pa
                data[chemical][variable] *= 6894.76
                units[variable] = '(Pa)'

            if read_units[variable].find('(deg F)') >= 0:
                # Convert to K
                data[chemical][variable] = (data[chemical][variable] - 32.) * 5 / 9 + 273.15
                units[variable] = '(K)'

            if read_units[variable].find('mol/dm^3 atm') >= 0:
                # Convert to kg/(m^3 Pa)
                data[chemical][variable] = (data[chemical][variable] * 1000. / 101325 * data[chemical]['M'])
                units[variable] = '(kg/(m^3 Pa))'

            if read_units[variable].find('mm^2/s') >= 0:
                # Convert to m^2/s
                data[chemical][variable] = data[chemical][variable] / 1000. ** 2
                units[variable] = '(m^2/s)'

            if read_units[variable].find('cal/mol') >= 0:
                # Convert to J/mol
                data[chemical][variable] /= 0.238846
                units[variable] = '(J/mol)'

            if read_units[variable].find('(g/cm^3)') >= 0.:
                # Convert to kg/m^3
                data[chemical][variable] = data[chemical][variable] / 1000 * 100 ** 3
                units[variable] = '(kg/m^3)'

    return data, units

