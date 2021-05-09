import datetime as datetime
import logging
import multiprocessing as mp
import sys
import time as time

import cirq as cirq
import numpy as numpy
import pandas as pandas
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4
from scipy.optimize import minimize

from vqe_functions import *


def calculate_minimum(molecule_name, basis, multiplicity, charge):
    """ Leiab miinimum energia molekuli tasakaalu olekule.

    Args:
        molecule_name (String): Molekuli keemiline lühend
        basis (int): Arvutuste baas
        multiplicity (int): Molekuli multiplicity
        charge (int): Molekuli laeng
    """
    logging.info("Calculating %s minimum.", molecule_name)

    # Molekulide tasakaalu olekute geomeetriad.
    min_geometry_dict = {'H2': [[ 'H', [ 0, 0, 0]],
                            [ 'H', [ 0, 0, 0.74]]], 
                    'LiH': [['Li', [0, 0, 0]] ,
                            ['H', [0, 0, 1.5949]]],
                    'BeH2': [['Be', [ 0, 0, 0 ]],
                            ['H', [ 0, 0, 1.3264]],
                            ['H', [ 0, 0, -1.3264]]],
                    'H2O': [['O', [-0.053670056908, -0.039737675589, 0]],
                            ['H', [-0.028413670411,  0.928922556351, 0]],
                            ['H', [0.880196420813,  -0.298256807934, 0]]]}
    geometry = min_geometry_dict[molecule_name]

    molecular_data = MolecularData(geometry, basis, multiplicity,
        charge, filename = './data/{}_min_molecule.data'.format(molecule_name))

    #Psi4 arvutused.
    molecular_data = run_psi4(molecular_data,
                            run_scf=True,
                            run_mp2=True,
                            run_cisd=True,
                            run_ccsd=True,
                            run_fci=True)

    # Miinimum väärtuse leidmine.
    file_name = "./results/VQE_min_{}_{}.csv".format(molecule_name, datetime.datetime.now())
    min_result = single_point_calculation([molecular_data, file_name])

    # Molekuli info logimine.
    logging.info("Electron count: %s", molecular_data.n_electrons)
    logging.info("Qubit count: %s", molecular_data.n_qubits)
    logging.info("Orbital count: %s", molecular_data.n_orbitals)
    logging.info("Results saved.")
    

def calculate_scan(molecule_name, basis, multiplicity, charge, counts):
    """ Leiab miinimum energia molekuli eri tuumade vahelistel kaugustel

    Args:
        molecule_name (String): Molekuli keemiline lühend
        basis (int): Arvutuste baas
        multiplicity (int): Molekuli multiplicity
        charge (int): Molekuli laeng
        counts (int): Arvutus punktide arv
    """
    
    length_bounds = [0.2, 3]
    logging.info("Calculating %s scan.", molecule_name)
    file_name = "./results/VQE_scan_{}_{}.csv".format(molecule_name, datetime.datetime.now())
    molecular_data_list = list()

    # Erinevate kauguste psi4 arvutuste läbi viimine enne VQE algoritmi rakendamist.
    for length in numpy.linspace(length_bounds[0], length_bounds[1], counts):
        length = round(length, 3)
        while True:
            geometry_dict = {'H2': [[ 'H', [ 0, 0, 0]],
                                [ 'H', [ 0, 0, length]]], 
                        'LiH': [['Li', [0, 0, 0]] ,
                                ['H', [0, 0,  length]]],
                        'BeH2': [['Be', [ 0, 0, 0 ]],
                                ['H', [ 0, 0,  length]],
                                ['H', [ 0, 0, - length]]],
                        'H2O': [['O', [0, 0, 0]],
                                ['H', [numpy.sin(0.9163) * length,  numpy.cos(0.9163) * length, 0]],
                                ['H', [- numpy.sin(0.9163) * length,  numpy.cos(0.9163) * length, 0]]]}

            geometry = geometry_dict[molecule_name]

            try:
                logging.info("Trying psi4 calculation at length %s.", length)
                molecular_data = MolecularData(geometry, basis, multiplicity,
                    charge, filename = './data/{}_{}_scan_molecule.data'.format(molecule_name, length), description=str(length))

                molecular_data = run_psi4(molecular_data,
                                        run_scf=True,
                                        run_mp2=True,
                                        run_cisd=True,
                                        run_ccsd=True,
                                        run_fci=True)
                
                logging.info("Psi4 calculations were succesful.")

                molecular_data_list.append([molecular_data, file_name])
                break
            except Exception as exc:
                # Kui antud molekuli geomeetriaga ei suutnud psi4 vajalike arvutusi teha,
                # suurendatakse tuumade vahelist vahekaugust 0.001 ja proovitakse uuesti.
                logging.error(exc)
                length += 0.001
                logging.info("New length set: %s", length)

    # Erinevate kauguste miinimumid arvutatakse paraleelselt.
    pool = mp.Pool(processes= 10)
    result = pool.map(single_point_calculation, molecular_data_list)


def main():
    """
    -min molecule \n
    -scan counts molecule \n
    Molecules: H2, LiH, BeH2
    """
    start = time.time()
    # Programmi argumentide sisse lugemine.
    args = sys.argv[1:]
    assert len(args) >= 2
    min_mode = ('-min' in args)
    scan_mode = ('-scan' in args)
    molecule_name = args[-1]
    if scan_mode:
        counts = int(args[-2])

    # Logimis faili üles seadmine.
    logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./logs/LOG_{}_{}.log".format(molecule_name, datetime.datetime.now())),
        logging.StreamHandler()
    ])

    # Kõigi arvutuste jaoks kehtivad järgmised tingimused.
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    
    # Viiakse läbi kas miinimum arvutus või mitme punkti arvutus.
    if min_mode:
        calculate_minimum(molecule_name, basis, multiplicity, charge)
    elif scan_mode:
        calculate_scan(molecule_name, basis, multiplicity, charge, counts)
    else:
        print("No mode selected!")
        exit()
    
    elapsed_time = time.time() - start
    logging.info("Total programm run time: %s s.", elapsed_time)
    exit()


# Promgramm algab:
if __name__ == "__main__":
    main()