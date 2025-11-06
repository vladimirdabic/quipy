from __future__ import annotations
from typing import List, Tuple, Iterable
from fractions import Fraction
import numpy as np
from .. import ops


class State:
    matrix: np.ndarray
    num_qubits: int
    latex_symbol: str = "\rho"

    def __init__(self, amplitudes: List[complex] = None, num_qubits: int = None,
                 density_matrix: np.ndarray = None, latex_symbol: str = r"\rho"):
        if amplitudes is not None:
            self.matrix = self.__prepare_state(amplitudes)
        elif num_qubits is not None:
            self.num_qubits = num_qubits
            state_vector = np.zeros(2 ** num_qubits, dtype=np.complex128)
            state_vector[0] = 1 # |00 .. 0>
            self.matrix = np.outer(state_vector, state_vector.conj())
        elif density_matrix is not None:
            self.num_qubits = int(np.log2(density_matrix.shape[0]))
            self.matrix = density_matrix
        else:
            raise ValueError("Must provide amplitudes, num_qubits or density_matrix")
        
        self.__generator = np.random.default_rng()
        self.latex_symbol = latex_symbol

    def __prepare_state(self, amplitudes: List[complex]):
        state = np.array(amplitudes, dtype=np.complex128)

        # Normalize if sum is not 1
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0):
            state = state / norm

        # Check dimension (must be 2^n)
        n_qubits = int(np.log2(len(state)))
        if 2**n_qubits != len(state):
            raise ValueError("Number of amplitudes must be 2^n for n qubits")
        
        self.num_qubits = n_qubits

        # Create a density matrix
        state2d = state[np.newaxis]
        col_vec = state2d.T
        return col_vec @ state2d.conj()

    def apply_gate(self, U: np.ndarray, qubit: int = None, qubits: List[int] = None):
        """
        Applies a gate to a qubit or qubits.

        Parameters
        ----------
        U : ndarray
            The unitary matrix representing the gate

        qubits : tuple of int
            Qubits to apply the gate to
        """
        if qubit is not None:
            qubits = [qubit]
        elif qubits is None:
            raise ValueError("Must provide qubit or qubits")
        
        if U is None:
            raise ValueError("Must provide gate matrix")

        operators = [ops.I] * self.num_qubits
        for qubit_idx in qubits:
            operators[qubit_idx] = U

        full_U = operators[0]
        for i in range(1, self.num_qubits):
            full_U = np.kron(full_U, operators[i])

        self.matrix = full_U @ self.matrix @ full_U.conj().T

    def apply_operator(self, U: np.ndarray):
        """
        Applies an operator to the whole state.

        Parameters
        ----------
        U : ndarray
            The matrix reprsenting the operator
        """
        if U is None:
            raise ValueError("Must provide operator matrix")

        self.matrix = U @ self.matrix @ U.conj().T

    def expectation_value(self, U: np.ndarray) -> np.complex128:
        """
        Returns the expectation value of an operator.

        Parameters
        ----------
        U : ndarray
            The operator
        
        Returns
        -------
        complex128
            The expectation value of U
        """
        if U is None:
            raise ValueError("Must provide operator matrix")

        return np.trace(U @ self.matrix)
    
    def probabilities(self) -> np.ndarray:
        """
        Returns the list of probabilities for the state (the diagonal of the density matrix).

        Returns
        -------
        ndarray
            The list of probabilities

        """
        return np.abs(self.matrix.diagonal())
    
    def partial_trace(self, keep_qubits: List[int]) -> State:
        """
        Returns a subsystem of the state (the partial trace).

        Parameters
        ----------
        keep_qubits: list of int
            The qubits that make up the subsystem

        Returns
        -------
        State
            The state of the subsystem
        """
        if keep_qubits is None:
            raise ValueError("Must provide qubits for the subsystem")

        trace_qubits = [i for i in range(self.num_qubits) if i not in keep_qubits]
        n_keep = len(keep_qubits)
        dim_keep = 2 ** n_keep
        
        s0 = np.array([[1], [0]], dtype=np.complex128)  # |0>
        s1 = np.array([[0], [1]], dtype=np.complex128)  # |1>

        # Basis states
        basis_states = []
        n_trace = len(trace_qubits)
        for i in range(2 ** n_trace):
            bin_digits = [int(c) for c in format(i, f"0{n_trace}b")]
            basis_states.append([s1 if b == 1 else s0 for b in bin_digits])

        # Apply basis states and sum them up (operator-sum representation)
        rho_new = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        for ops_list in basis_states:
            basis_op = ops.build_operators(self.num_qubits, qubits=trace_qubits, operators=ops_list)
            rho_new += basis_op.conj().T @ self.matrix @ basis_op
                
        return State(density_matrix=rho_new)

    def subsystem(self, qubit: int) -> State:
        """
        Returns the subsystem for a singular qubit.

        Parameters
        ----------
        qubit : int
            The qubit

        Returns
        -------
            The state of the qubit
        """
        return self.partial_trace(keep_qubits=[qubit])
    
    def measure(self, qubit: int = None, qubits: List[int] = None) -> List[int]:
        """
        Performs a projective measurement on the qubit or qubits.

        Parameters
        ----------
        qubit : int, optional
            The qubit to measure

        qubits : list of int, optional
            The qubits to measure

        Returns
        -------
        list of int
            The list of measurement results
        """
        if qubit is not None:
            qubits = [qubit]
        elif qubits is None:
            raise ValueError("Must provide qubit or qubits")

        sub = self.partial_trace(keep_qubits=qubits)
        probs = sub.probabilities()

        idx = self.__generator.choice(len(probs), p=probs)
        prob = probs[idx]
        results = [int(c) for c in format(idx, f"0{sub.num_qubits}b")]

        # Collapse
        proj_op = ops.build_operators(self.num_qubits, qubits=qubits, operators=[ops.P1 if b == 1 else ops.P0 for b in results])
        rho_next = (proj_op @ self.matrix @ proj_op) / prob

        # Renormalize
        norm = np.linalg.norm(rho_next)
        if not np.isclose(norm, 1.0):
            rho_next = rho_next / norm

        self.matrix = rho_next
        return results
    
    def copy(self) -> State:
        """
        Creates a copy of the state

        Returns
        -------
        State
            The copy
        """
        return State(density_matrix=self.matrix.copy(), latex_symbol=self.latex_symbol)
    

    # Latex stuff
    def to_latex(self, threshold: float = 1e-10, max_terms: int = 20, use_fractions: bool = True, include_symbol: bool = True) -> str:
        """
        Convert to LaTeX representation.
        
        Parameters
        ----------
        threshold : float
            Ignore terms with |coeff| < threshold
        max_terms : int
            Maximum number of terms to display
        use_fractions : bool
            Try to represent decimals as fractions

        Returns
        -------
        str
            The latex representation of the state
        """
        latex_terms = []
        
        for ket in range(1 << self.num_qubits):
            for bra in range(1 << self.num_qubits):
                coeff = self.matrix[ket, bra]

                if abs(coeff) < threshold:
                    continue
                
                coeff_str = self._format_coefficient(coeff, use_fractions)
                ket_str = format(ket, f'0{self.num_qubits}b')
                bra_str = format(bra, f'0{self.num_qubits}b')
                
                # Build term
                term = f"{coeff_str}|{ket_str}\\rangle\\langle {bra_str}|"
                latex_terms.append(term)
                
                if len(latex_terms) >= max_terms:
                    latex_terms.append(r"\cdots")
                    break
            
            if len(latex_terms) >= max_terms:
                break
        
        # Join with + signs
        result = (f"{self.latex_symbol} = " if include_symbol else "") + " + ".join(latex_terms)
        result = result.replace("+ -", "- ")
        
        return result
    
    def _format_coefficient(self, coeff, use_fractions=True):
        if abs(coeff.imag) < 1e-10:
            # Real number
            real = coeff.real
            
            if use_fractions:
                frac_str = self._try_fraction(real)
                if frac_str:
                    return frac_str
            
            # Otherwise format as decimal
            if abs(real - 1.0) < 1e-10:
                return ""  # Omit coefficient of 1
            elif abs(real + 1.0) < 1e-10:
                return "-"  # Just minus sign
            else:
                return f"{real:.3f}"
        
        else:
            # Complex number
            real = coeff.real
            imag = coeff.imag
            
            parts = []
            if abs(real) > 1e-10:
                parts.append(f"{real:.3f}")
            if abs(imag) > 1e-10:
                if imag > 0 and parts:
                    parts.append(f"+{imag:.3f}i")
                else:
                    parts.append(f"{imag:.3f}i")
            
            return f"({' '.join(parts)})"
    
    def _try_fraction(self, value, max_denominator=16):
        try:
            frac = Fraction(value).limit_denominator(max_denominator)
            
            # Only use fraction if it's close enough
            if abs(float(frac) - value) < 1e-10:
                if frac.denominator == 1:
                    if frac.numerator == 1:
                        return ""  # Omit coefficient of 1
                    elif frac.numerator == -1:
                        return "-"
                    else:
                        return str(frac.numerator)
                else:
                    return f"\\frac{{{frac.numerator}}}{{{frac.denominator}}}"
        except:
            pass
        
        return None
    
    # For jupyter
    def _repr_latex_(self):
        return f"$${self.to_latex()}$$"
    
    def __repr__(self):
        return self.to_latex()