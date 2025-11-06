from __future__ import annotations
from typing import Dict, Tuple, List
from enum import IntEnum, auto
from fractions import Fraction
import numpy as np

class BitOrder(IntEnum):
    LittleEndian = auto()
    BigEndian = auto()

class State:
    terms: Dict[Tuple[int], np.complex128]
    num_qubits: int
    bit_order: BitOrder
    latex_symbol: str

    def __init__(self, num_qubits: int = None, amplitudes: List[complex] = None,
                 bit_order: BitOrder = BitOrder.BigEndian, latex_symbol: str = r"\rho",
                 no_terms: bool = False):
        self.terms = {}
        self.bit_order = bit_order
        self.latex_symbol = latex_symbol
        self.__generator = np.random.default_rng()
        
        if num_qubits is not None:
            self.num_qubits = num_qubits
            if not no_terms: self.add_term(0, 0, 1)
        elif amplitudes is not None:
            self.__prepare_state(amplitudes)
        else:
            raise ValueError("Must provide num_qubits or list of amplitudes")

    def add_term(self, ket: int, bra: int, coeff: complex):
        """
        Adds a density matrix entry (term) to the state.

        Parameters
        ----------
        ket : int
            Binary value representing the state of the ket (row)

        bra : int
            Binary value representing the state of the bra (column)

        coeff : int
            The coefficient 
        """
        key = (ket, bra)
        coeff = np.complex128(coeff)

        if key in self.terms:
            self.terms[key] += coeff
        else:
            self.terms[key] = coeff

        if(abs(self.terms[key])) < 1e-10:
            del self.terms[key]

    def apply_gate(self, U: np.ndarray, qubit: int = None, qubits: List[int] = None):
        """
        Applies a gate to a qubit or qubits.

        Parameters
        ----------
        U : ndarray
            The unitary matrix representing the gate

        qubit : int, optional
            The qubit to apply the gate to

        qubits : tuple of int, optional
            Qubits to apply the gate to
        """
        self.apply_sides(U, U, qubit=qubit, qubits=qubits)

    def apply_sides(self, U_left: np.ndarray, U_right: np.ndarray, qubit: int = None, qubits: List[int] = None):
        """
        Applies two different gates to the kets and bras of a qubit or qubits.

        Parameters
        ----------
        U_left : ndarray
            The unitary matrix representing the left gate

        U_right : ndarray
            The unitary matrix representing the right gate

        qubit : int, optional
            The qubit to apply the gate to

        qubits : tuple of int, optional
            Qubits to apply the gate to
        """
        if qubit is not None:
            qubits = [qubit]
        elif qubits is None:
            raise ValueError("Must provide qubit or qubits")
        
        if U_left is None or U_right is None:
            raise ValueError("Must provide gate matrix")
        
        U_herm = U_right.conj().T

        for q in qubits:
            new_terms = {}

            for (ket, bra), coeff in self.terms.items():
                if q >= self.num_qubits:
                    raise IndexError(f"Qubit is out of range (Q{q} >= #{self.num_qubits})")

                new_ket_terms = self.__apply(U_left, ket, q)
                new_bra_terms = self.__apply(U_herm, bra, q)

                for ket_new, ket_amp in new_ket_terms.items():
                    for bra_new, bra_amp in new_bra_terms.items():
                        new_coeff = coeff * ket_amp * bra_amp.conj()
                        key = (ket_new, bra_new)
                        new_terms[key] = new_terms.get(key, 0) + new_coeff
        
            self.terms = new_terms
            self._cleanup()

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

        new_terms = {}
        U_herm = U.conj().T

        for (ket, bra), coeff in self.terms.items():
            new_ket_terms = self.__apply_whole(U, ket)
            new_bra_terms = self.__apply_whole(U_herm, bra)

            for ket_new, ket_amp in new_ket_terms.items():
                for bra_new, bra_amp in new_bra_terms.items():
                    new_coeff = coeff * ket_amp * bra_amp.conj()
                    key = (ket_new, bra_new)
                    new_terms[key] = new_terms.get(key, 0) + new_coeff
        
        self.terms = new_terms
        self._cleanup()

    def apply_composite_operator(self, *terms):
        """
        Apply composite operator: sum of tensor products.
        
        Parameters
        ----------
        *terms : list of (operator_list, qubit_list) tuples
            e.g., ([P0, I], [0, 1]), ([P1, X], [0, 1])
        """
        new_state = State(num_qubits=self.num_qubits, bit_order=self.bit_order, latex_symbol=self.latex_symbol, no_terms=True)

        for (ops_L, qubits_L) in terms:
            for (ops_R, qubits_R) in terms:
                temp_state = self.copy()

                for opr, qr, opl, ql in zip(ops_R, qubits_R, ops_L, qubits_L):
                    temp_state.apply_sides(opl, opr, qubit=ql)

                new_state += temp_state
        
        self.terms = new_state.terms
        self._cleanup()

    def expectation_value(self, U: np.ndarray, qubit: int = None) -> np.complex128:
        """
        Returns the expectation value of an operator.

        Parameters
        ----------
        U : ndarray
            The operator

        qubit : int, optional
            The qubit the operator acts on
        
        Returns
        -------
        complex128
            The expectation value of U
        """
        if U is None:
            raise ValueError("Must provide operator matrix")

        result = 0

        for (ket, bra), coeff in self.terms.items():
            # Compute <bra| U |ket>
            ket_after_O = self.__apply_whole(U, ket) if qubit is None else self.__apply(U, ket, qubit)
            
            for new_ket, amplitude in ket_after_O.items():
                if new_ket == bra: # Orthonormality
                    result += coeff * amplitude
    
        return result.real
    
    def probabilities(self) -> np.ndarray:
        """
        Returns the list of probabilities for the state (the diagonal of the density matrix).

        Returns
        -------
        ndarray
            The list of probabilities
        """
        return np.abs([self.terms.get((i, i), 0) for i in range(1 << self.num_qubits)])
    
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
        
        positions = sorted([self.__get_qubit_position(q) for q in keep_qubits])
        n_keep = len(positions)
        
        subsystem = State(num_qubits=n_keep, bit_order=self.bit_order, latex_symbol=self.latex_symbol)
        new_terms = {}

        for (ket, bra), coeff in self.terms.items():
            ket_new = self.__pack_bits(ket, positions)
            bra_new = self.__pack_bits(bra, positions)

            # Only include terms where traced-out bits are equal (orthonormality)
            if all(((ket >> q) & 1) == ((bra >> q) & 1) for q in range(self.num_qubits) if q not in positions):
                new_terms[(ket_new, bra_new)] = new_terms.get((ket_new, bra_new), 0) + coeff

        subsystem.terms = new_terms
        subsystem._cleanup()
        return subsystem

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
        new_terms = {}
        positions = [self.__get_qubit_position(q) for q in qubits]

        for (ket, bra), coeff in self.terms.items():
            # If bits match the results (projecting)
            if all(((ket >> pos) & 1) == res and ((bra >> pos) & 1) == res for pos, res in zip(positions, results)):
                new_terms[(ket, bra)] = coeff / prob # Normalized 

        self.terms = new_terms
        self._cleanup()
        return results

    def copy(self) -> State:
        """
        Creates a copy of the state

        Returns
        -------
        State
            The copy
        """
        copied = State(num_qubits=self.num_qubits, bit_order=self.bit_order)
        copied.terms = self.terms.copy()
        return copied
    
    # Helper function
    def __pack_bits(self, num: int, positions: List[int]) -> int:
        packed = 0
        
        for shift, pos in enumerate(positions):
            packed |= ((num >> pos) & 1) << shift

        return packed

    def __apply(self, U: np.ndarray, ket: int, qubit: int) -> Dict[int, np.complex128]:
        result = {}
        actual_q = self.__get_qubit_position(qubit)
        target_ket = (ket >> actual_q) & 1
        
        for output_ket in [0, 1]:
            amplitude = U[output_ket, target_ket]
            if np.abs(amplitude) < 1e-10:
                continue

            new_ket = ket & ~(1 << actual_q)  # Clear bit
            new_ket |= (output_ket << actual_q)  # Set bit
            
            result[new_ket] = amplitude

        return result
    
    def __apply_whole(self, U: np.ndarray, ket: int) -> Dict[int, np.complex128]:
        result = {}
    
        for output_ket in range(1 << self.num_qubits):
            amplitude = U[output_ket, ket]
            if np.abs(amplitude) < 1e-10:
                continue
            
            result[output_ket] = amplitude
        
        return result
    
    def _cleanup(self):
        self.terms = {k: v for k, v in self.terms.items() if abs(v) > 1e-10}

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

        for ket, ket_amp in enumerate(state):
            for bra, bra_amp in enumerate(state):
                self.add_term(ket, bra, ket_amp * bra_amp.conj())

    def __get_qubit_position(self, qubit: int) -> int:
        if qubit >= self.num_qubits:
            raise IndexError(f"Qubit is out of range (Q{qubit} >= #{self.num_qubits})")

        if self.bit_order == BitOrder.BigEndian:
            return self.num_qubits - qubit - 1
        
        return qubit
    
    def __add__(self, other):
        if type(other) != State:
            raise TypeError(f"Cannot add State and {type(other)}")

        result = self.copy()
        
        for key, val in other.terms.items():
            result.terms[key] = result.terms.get(key, 0) + val
        
        result._cleanup()
        return result
    

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
        if not self.terms:
            return f"{self.latex_symbol} = 0" if include_symbol else "0"
        
        latex_terms = []
        
        for (ket, bra), coeff in sorted(self.terms.items()):
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