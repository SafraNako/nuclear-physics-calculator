"""Radioactive decay chains.

A chain 1 -> 2 -> ... -> n with decay constants lambda_i and branching fractions
b_i (the fraction of decays of member i that feed member i+1) obeys the linear
system dN/dt = M N, with M_ii = -lambda_i and M_{i+1,i} = b_i*lambda_i.

chain_populations() solves it with the matrix exponential, N(t) = expm(M t) N(0).
That is exact for any set of decay constants, equal ones included, and handles a
stable end member (lambda = 0).

bateman_closed_form() evaluates the textbook Bateman solution

    N_k(t) = N_1(0) * prod_{i<k} b_i lambda_i * sum_{i<=k} exp(-lambda_i t) / prod_{j<=k, j!=i} (lambda_j - lambda_i)

It needs all lambda distinct and loses precision in two situations, both because
the terms are individually large and cancel: when two decay constants approach
each other (relative error grows like eps_machine / |lambda_j - lambda_i|: 9e-14,
7e-11, 1.2e-7, 1.1e-4 for relative separations 1e-3, 1e-6, 1e-9, 1e-12), and for
deep chain members at early times, when the true value is tiny (3.5e-11 for the
fourth member of a 4-chain at t = 1). bateman_reference() evaluates the same
formula with 80-digit decimal arithmetic and serves as the truth in the tests.

Two closed-form consequences for a parent (1) and daughter (2) with
lambda_1 < lambda_2 (transient equilibrium), unit branching:

    A_2 / A_1  ->  lambda_2 / (lambda_2 - lambda_1)  =  T_1 / (T_1 - T_2)
    t_max      =  ln(lambda_2 / lambda_1) / (lambda_2 - lambda_1)       (daughter activity peaks)
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext

import numpy as np
from scipy.linalg import expm

from .decay import decay_constant


def _lambdas(half_lives_s) -> list:
    return [decay_constant(t) for t in half_lives_s]


def _branching(n: int, branching):
    if branching is None:
        return [1.0] * (n - 1)
    if len(branching) != n - 1:
        raise ValueError(f"{n} üyeli zincir için {n - 1} dallanma oranı gerekli")
    if any(not 0.0 <= b <= 1.0 for b in branching):
        raise ValueError("dallanma oranları [0, 1] aralığında olmalı")
    return list(branching)


def decay_matrix(half_lives_s, branching=None) -> np.ndarray:
    lam = _lambdas(half_lives_s)
    n = len(lam)
    b = _branching(n, branching)
    m = np.zeros((n, n))
    for i in range(n):
        m[i, i] = -lam[i]
        if i < n - 1:
            m[i + 1, i] = b[i] * lam[i]
    return m


def chain_populations(half_lives_s, times_s, branching=None, n0: float = 1.0) -> np.ndarray:
    """N_i(t) for each time (rows) and chain member (columns); only member 1 is present at t = 0."""
    m = decay_matrix(half_lives_s, branching)
    start = np.zeros(len(m))
    start[0] = n0
    return np.array([expm(m * float(t)) @ start for t in times_s])


def chain_activities(half_lives_s, times_s, branching=None, a0: float = 1.0) -> np.ndarray:
    """Activities lambda_i*N_i(t) given the parent's initial activity a0."""
    lam = np.array(_lambdas(half_lives_s))
    parent_lambda = lam[0]
    if parent_lambda == 0.0:
        raise ValueError("ana çekirdek kararlıysa zincir bozunmaz")
    return chain_populations(half_lives_s, times_s, branching, n0=a0 / parent_lambda) * lam


def bateman_closed_form(half_lives_s, t_s: float, branching=None, n0: float = 1.0) -> np.ndarray:
    lam = _lambdas(half_lives_s)
    n = len(lam)
    b = _branching(n, branching)
    out = []
    for k in range(1, n + 1):
        prefactor = n0 * math.prod(b[i] * lam[i] for i in range(k - 1))
        total = 0.0
        for i in range(k):
            denominator = math.prod(lam[j] - lam[i] for j in range(k) if j != i)
            total += math.exp(-lam[i] * t_s) / denominator
        out.append(prefactor * total)
    return np.array(out)


def bateman_reference(half_lives_s, t_s: float, branching=None, n0: float = 1.0, digits: int = 80) -> list:
    """Bateman solution in `digits`-digit decimal arithmetic (list of Decimal)."""
    getcontext().prec = digits
    two_ln = Decimal(2).ln()
    lam = [two_ln / Decimal(repr(float(t))) for t in half_lives_s]
    n = len(lam)
    b = [Decimal(repr(float(x))) for x in _branching(n, branching)]
    t = Decimal(repr(float(t_s)))
    out = []
    for k in range(1, n + 1):
        prefactor = Decimal(repr(float(n0)))
        for i in range(k - 1):
            prefactor *= b[i] * lam[i]
        total = Decimal(0)
        for i in range(k):
            denominator = Decimal(1)
            for j in range(k):
                if j != i:
                    denominator *= lam[j] - lam[i]
            total += (-lam[i] * t).exp() / denominator
        out.append(prefactor * total)
    return out


def transient_equilibrium_ratio(parent_half_life_s: float, daughter_half_life_s: float, branching: float = 1.0) -> float:
    """A_daughter / A_parent at large t; requires the parent to be the longer-lived one."""
    if parent_half_life_s <= daughter_half_life_s:
        raise ValueError("geçici denge için ana çekirdek daha uzun ömürlü olmalı")
    return branching * parent_half_life_s / (parent_half_life_s - daughter_half_life_s)


def time_of_maximum_daughter_activity(parent_half_life_s: float, daughter_half_life_s: float) -> float:
    l1, l2 = decay_constant(parent_half_life_s), decay_constant(daughter_half_life_s)
    if l1 == l2:
        return 1.0 / l1
    return math.log(l2 / l1) / (l2 - l1)
