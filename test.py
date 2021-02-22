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

qubit_operator = ((QubitOperator('X2 X3 Y10 X11', -0.125j)
+ 0.125j * QubitOperator('Y2 X3 X10 X11')
+ 0.125j * QubitOperator('Y2 Y3 Y10 X11')
+ 0.125j * QubitOperator('X2 Y3 X10 X11')
- 0.125j * QubitOperator('Y2 X3 Y10 Y11')
- 0.125j * QubitOperator('X2 X3 X10 Y11')
- 0.125j * QubitOperator('X2 Y3 Y10 Y11')
+ 0.125j * QubitOperator('Y2 Y3 X10 Y11')))
qubit_operator_list = [qubit_operator]

qubit_count = 14
test_circuit = create_UCCSD(qubit_operator_list, qubit_count, 't')

qubits = cirq.LineQubit.range(qubit_count)
moment = cirq.Moment()
for qubit in qubits:
    moment = moment.with_operation(cirq.measure(qubit))
test_circuit.append(moment)

print(test_circuit.to_text_diagram(transpose=True))