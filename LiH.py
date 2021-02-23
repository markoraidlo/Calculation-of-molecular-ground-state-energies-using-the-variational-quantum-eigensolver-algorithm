import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
from openfermion import MolecularData
from openfermionpsi4 import run_psi4
import time

from functions import *

start = time.process_time()
molecule_name = "LiH"
geometry= [['Li', [0, 0, 0]] ,
                ['H', [0, 0, 1.5949]]]
basis = 'sto-3g'
multiplicity = 1
charge = 0

molecular_data = MolecularData(geometry, basis, multiplicity,
    charge, filename = './data/{}_molecule.data'.format(molecule_name))

molecular_data = run_psi4(molecular_data,
                        run_scf=True,
                        run_mp2=True,
                        run_cisd=True,
                        run_ccsd=True,
                        run_fci=True)

qubit_operator_list = get_qubit_operators(molecular_data)

electron_count = molecular_data.n_electrons
qubit_count = molecular_data.n_qubits

UCCSD = initial_hartree_fock(electron_count)
UNITARY = create_UCCSD(qubit_operator_list, qubit_count, 't')
UCCSD.append(UNITARY, strategy = cirq.InsertStrategy.NEW)

hamiltonian = get_measurement_hamiltonian(molecular_data)
coefficients, measurement_circuits = get_measurement_circuits(hamiltonian, qubit_count)

elapsed = time.process_time() - start

print(molecule_name)
print("Electron count: {}".format(electron_count))
print("Qubit count: {}".format(qubit_count))
print("Qubit operator count: {}".format(len(qubit_operator_list)))
print("Length of UCCSD circuit: {}".format(len(UCCSD)))
print("Number of measurement circuits: {}".format(len(measurement_circuits)))
print("Time elapsed: {}".format(elapsed))
#print(UCCSD.to_text_diagram(transpose=True))