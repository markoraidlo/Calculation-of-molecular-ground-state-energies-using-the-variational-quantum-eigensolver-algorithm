import cirq as cirq
import numpy as numpy
import qsimcirq as qsimcirq
import sympy as sympy
from openfermion import (FermionOperator, MolecularData, bravyi_kitaev,
                         get_fermion_operator, jordan_wigner,
                         uccsd_convert_amplitude_format)
from openfermionpsi4 import run_psi4

from openfermion import QubitOperator

from functions import *

#TODO: Code the test

#Mingi qubit
qubit_operator = QubitOperator()
qubit_count = 10
test_circuit = create_UCCSD(qubit_operator, qubit_count, 't')
print(test_circuit.to_text_diagram(transpose=True))