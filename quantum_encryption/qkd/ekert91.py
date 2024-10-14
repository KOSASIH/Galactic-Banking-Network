# ekert91.py

import numpy as np
from qiskit import QuantumCircuit, execute, Aer

def ekert91_protocol(alice_bits, bob_basis):
    """
    Simulate the Ekert91 QKD protocol.

    Parameters:
    - alice_bits: Alice's random bits.
    - bob_basis: Bob's random basis.

    Returns:
    - alice_key: Alice's key.
    - bob_key: Bob's key.
    """
    # Create a quantum circuit
    circuit = QuantumCircuit(2)

    # Prepare Alice's qubits
    for i, bit in enumerate(alice_bits):
        if bit == 0:
            circuit.ry(np.pi / 2, 0)
        else:
            circuit.ry(-np.pi / 2, 0)

    # Prepare Bob's qubits
    for i, basis in enumerate(bob_basis):
        if basis == 0:
            circuit.ry(np.pi / 2, 1)
        else:
            circuit.ry(-np.pi / 2, 1)

    # Measure the qubits
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1024)
    result = job.result()

    # Extract the measurement outcomes
    outcomes = result.get_counts(circuit)

    # Post-processing
    alice_key = []
    bob_key = []
    for outcome in outcomes:
        if outcome == '00':
            alice_key.append(0)
            bob_key.append(0)
        elif outcome == '11':
            alice_key.append(1)
            bob_key.append(1)

    return alice_key, bob_key
