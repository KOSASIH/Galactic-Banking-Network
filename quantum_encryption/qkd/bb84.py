# bb84.py

import numpy as np
from qiskit import QuantumCircuit, execute, Aer

def bb84_protocol(alice_bits, bob_basis):
    """
    Simulate the BB84 QKD protocol.

    Parameters:
    - alice_bits: Alice's random bits.
    - bob_basis: Bob's random basis.

    Returns:
    - alice_key: Alice's key.
    - bob_key: Bob's key.
    """
    # Create a quantum circuit
    circuit = QuantumCircuit(1)

    # Prepare Alice's qubits
    for i, bit in enumerate(alice_bits):
        if bit == 0:
            circuit.ry(np.pi / 2, 0)
        else:
            circuit.ry(-np.pi / 2, 0)

    # Measure Bob's qubits
    for i, basis in enumerate(bob_basis):
        if basis == 0:
            circuit.measure(0, 0)
        else:
            circuit.ry(np.pi / 2, 0)
            circuit.measure(0, 0)

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
        if outcome == '0':
            alice_key.append(0)
            bob_key.append(0)
        elif outcome == '1':
            alice_key.append(1)
            bob_key.append(1)

    return alice_key, bob_key
