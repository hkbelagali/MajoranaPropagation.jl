"""Utility functions for generating ansatze, and integrating with the rest of the codebase."""
from typing import Dict
from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW
from ffsim.linalg import givens_decomposition


def to_json(hf: PrepareHartreeFockJW, ucj: UCJOpSpinBalancedJW,
            time: float = -1.0) -> Dict:
    """
     Convert the Hartree-Fock state and LUCJ ansatz to a JSON-serializable format

     Parameters:
        hf: PrepareHartreeFockJW
         The Hartree-Fock state preparation circuit

         ucj: UCJOpSpinBalancedJW
            The LUCJ ansatz circuit

        time: float
            Imaginary time value to embed in the payload (default -1.0).

    Returns:
        A dictionary containing the JSON-serializable representations
        of the Hartree-Fock state and LUCJ ansatz. Use json.dumps()
        to write the dictionary to a JSON file.
    """

    def cx_to_pair(z):
        z = complex(z)
        return [float(z.real), float(z.imag)]

    def decompose_unitary(U):
        givens_rots, phase_shifts = givens_decomposition(U)
        rots_serialized = []
        for g in givens_rots:
            c, s, i, j = g
            rots_serialized.append({
                "c":    float(c.real) if hasattr(c, "real") else float(c),
                "s_re": float(complex(s).real),
                "s_im": float(complex(s).imag),
                "i":    int(i),
                "j":    int(j),
            })
        return {"givens": rots_serialized, "phase_shifts": [cx_to_pair(p) for p in phase_shifts]}

    norb = hf.norb
    nelec_a, nelec_b = hf.nelec
    nelectron = nelec_a + nelec_b

    ucj_op = ucj.ucj_op

    layers = []
    for orb_rot, (diag_mat_aa, diag_mat_ab) in zip(ucj_op.orbital_rotations, ucj_op.diag_coulomb_mats):
        layers.append({
            "fwd":         decompose_unitary(orb_rot.T.conj()),
            "inv":         decompose_unitary(orb_rot),
            "diag_mat_aa": [[float(x) for x in row] for row in diag_mat_aa],
            "diag_mat_ab": [[float(x) for x in row] for row in diag_mat_ab],
        })

    final_serialized = None
    if ucj_op.final_orbital_rotation is not None:
        final_serialized = decompose_unitary(ucj_op.final_orbital_rotation)

    return {
        "norb":      norb,
        "time":      float(time),
        "nelectron": nelectron,
        "layers":    layers,
        "final":     final_serialized,
    }

if __name__ == "__main__": 
    import itertools
    import math
    import cmath

    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    import ffsim
    import pyscf
    from pyscf import gto, scf, cc
    import json
    from ffsim.linalg import givens_decomposition
    from qiskit_ibm_runtime import  SamplerV2 as Sampler
    from qiskit.circuit.library import XXPlusYYGate, RYGate, RZZGate, CPhaseGate

    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    def generate_hchain_geometry(natoms: int, atomic_distance: float = 0.7) -> str:
        """
        Returns a linear Hydrogen chain geometry for use in PySCF molecule construction.
        
        Args:
            natoms: Number of Hydrogen atoms in the chain.
            atomic_distance: Equal spacing between Hydrogen atoms.
        """
        return "; ".join([f"H 0 0 {i * atomic_distance}" for i in range(natoms)])
    
    geom = generate_hchain_geometry(natoms=8)
    print(geom)

    mol = gto.Mole(atom = geom, charge = 0, basis = 'sto3g')
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    cc_ = cc.CCSD(mf).run()
    N = mol.nao_nr() * 2

    # Define active space
    n_frozen = 0
    active_space = range(n_frozen, mol.nao_nr())


    # Get molecular integrals
    scf = pyscf.scf.RHF(mol).run()
    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)
    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

    # Compute exact energy using FCI
    reference_energy = cas.run().e_tot

    print(f"norb = {norb}")
    print(f"nelec = {nelec}")

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    t1 = ccsd.t1
    t2 = ccsd.t2

    # Set ansatz properties
    n_reps = 2
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]  # None  # Let generate_lucj_pass_manager determine the alpha-beta interactions

    # Create the LUCJ ansatz operator
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab),
        # Setting optimize=True enables the "compressed" factorization
        optimize=True,
        # Limit the number of optimization iterations to prevent the code cell from running
        # too long. Removing this line may improve results.
        options=dict(maxiter=1000),
    )

    hf = ffsim.qiskit.PrepareHartreeFockJW(norb, nelec)
    lucj = ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op)

    payload = to_json(hf, lucj) 
    with open("test_payload.json", "w") as f:
        json.dump(payload, f, indent=2)