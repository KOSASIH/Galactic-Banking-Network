# quantum_key_exchange.py

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer.noise import depolarizing_error, NoiseModel
from qiskit.tools.monitor import job_monitor

def quantum_key_exchange(alice_bits, bob_basis, error_rate=0.01):
    """
    Simulate a quantum key exchange protocol using the E91 protocol.

    Parameters:
    - alice_bits: Alice's random bits.
    - bob_basis: Bob's random basis.
    - error_rate: Error rate of the quantum channel.

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

    # Add noise to the quantum channel
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
    circuit = noise_model.add_noise(circuit)

    # Measure the qubits
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1024)
    job_monitor(job)
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
