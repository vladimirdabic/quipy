# qilb

**qilb** is a quantum circuit simulator with sparse and dense density matrix implementations.


### qilb.dense
The dense simulation consists of direct numpy matrix manipulation, such as applying operators to the density matrix $U \rho U^\dagger$, taking the trace $\text{Tr}(\rho)$, etc. 

This model of simulation becomes inefficient when simulating large systems with a few matrix entries. For example, a 10 qubit system has a density matrix of size $1024 \times 1024$, which is $1048576$ elements! For a state, such as $\frac{1}{2}\ket{0}^{\otimes 10} \bra{0}^{\otimes 10}  + \frac{1}{2} \ket{1}^{\otimes 10}\bra{1}^{\otimes 10}$, this type of simulation is overkill.

### qilb.sparse
The sparse simulation consists of term manipulation. Internally, we treat the density matrix entries as a list of terms with coefficients, just like the mathematical definition of a density matrix:

$$ \rho = \sum_i p_i \ket{\psi_i} \bra{\psi_i}. $$

Instead of directly multiplying operators with the density matrix, we use the fact that they're linear, which means they can act on each term separately:

$$ U \rho U^\dagger = \sum_i p_i U\ket{\psi_i} \bra{\psi_i}U^\dagger $$

This type of simulation, as the name suggests, is preferred for sparse systems—systems with a small number of entries.

The state $\frac{1}{2}\ket{0}^{\otimes 10} \bra{0}^{\otimes 10}  + \frac{1}{2} \ket{1}^{\otimes 10}\bra{1}^{\otimes 10}$ in this simulation just requires two entries in the term list, compared to the $1048576$ elements required in the dense simulation.

### qilb.ops
This is an auxiliary module which holds definitions for common quantum gates, such as $I, X, Y, Z, H, S, T, R_x, R_y,$ and $R_z$.

# Examples
The library supports latex representation of the current state, which is useful in a jupyter notebook. Below is an example.


```python
import qilb.sparse as qis

state = qis.State(num_qubits=2)
state
```




$$\rho = |00\rangle\langle 00|$$



Below are sparse and dense examples of the Bell circuit.


```python
import qilb.sparse as qis
import qilb as qi

# Bell circuit
circuit = qis.Circuit(num_qubits=2)
circuit.add_gate(qi.ops.H, qubit=0)
circuit.CNOT(control_qubit=0, target_qubit=1)

# Running the circuit
state = qis.State(num_qubits=2)
final = circuit.run(initial_state=state)

final
```




$$\rho = \frac{1}{2}|00\rangle\langle 00| + \frac{1}{2}|00\rangle\langle 11| + \frac{1}{2}|11\rangle\langle 00| + \frac{1}{2}|11\rangle\langle 11|$$




```python
import qilb.dense as qis
import qilb as qi

# Bell circuit
circuit = qis.Circuit(num_qubits=2)
circuit.add_gate(qi.ops.H, qubit=0)
circuit.CNOT(control_qubit=0, target_qubit=1)

# Running the circuit
state = qis.State(num_qubits=2)
final = circuit.run(initial_state=state)

final
```




$$\rho = \frac{1}{2}|00\rangle\langle 00| + \frac{1}{2}|00\rangle\langle 11| + \frac{1}{2}|11\rangle\langle 00| + \frac{1}{2}|11\rangle\langle 11|$$



Below is an implementation of the repetition error correcting circuit using the sparse simulator.


```python
import qilb.sparse as qis
import qilb as qi
from IPython.display import HTML

circuit = qis.Circuit(num_qubits=5)

# Encoding
circuit.CNOT(control_qubit=0, target_qubit=1)
circuit.CNOT(control_qubit=0, target_qubit=2)

# Noise happens here
# Emulate a fake bitflip
circuit.add_gate(qi.ops.X, qubit=1)
circuit.add_callback(lambda state: display(HTML("State after a bit-flip error:"), state))

# Correction
def correct(state: qis.State, results):
    match results:
        case [1, 0]:
            state.apply_gate(qi.ops.X, qubit=0)
            display(HTML("Corrected qubit 0"))
        case [1, 1]:
            state.apply_gate(qi.ops.X, qubit=1)
            display(HTML("Corrected qubit 1"))
        case [0, 1]:
            state.apply_gate(qi.ops.X, qubit=2)
            display(HTML("Corrected qubit 2"))

circuit.CNOT(control_qubit=0, target_qubit=3)
circuit.CNOT(control_qubit=1, target_qubit=3)
circuit.CNOT(control_qubit=1, target_qubit=4)
circuit.CNOT(control_qubit=2, target_qubit=4)
circuit.measuring_device(qubits=[3, 4], callback=correct)


# Run
state = qis.State(num_qubits=5)
display(state)

final = circuit.run(initial_state=state)
final.latex_symbol = r"\rho^\prime"
display(final)
```


$$\rho = |00000\rangle\langle 00000|$$



State after a bit-flip error:



$$\rho = |01000\rangle\langle 01000|$$



Corrected qubit 1



$$\rho^\prime = |00011\rangle\langle 00011|$$

