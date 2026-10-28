"""
game_theory.py
Core mathematical and game-theoretic calculations for cloud resource allocation.

Handles:
- Resource allocation computation under capacity constraints (proportional scaling policy).
- Parametric utility / payoff calculation based on benefit, cost, congestion, and shortfall.
- Payoff matrix generation for 2-player strategic games.
- Pure-strategy best responses computation (including tie handling).
- Pure-strategy Nash equilibrium (PSNE) detection.
- Social welfare (utilitarian sum) and Pareto optimality evaluation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

STRATEGY_LOW = "Low Demand"
STRATEGY_HIGH = "High Demand"
STRATEGIES = [STRATEGY_LOW, STRATEGY_HIGH]


@dataclass
class AllocationResult:
    """Stores the resource allocation breakdown for a single strategy profile."""
    capacity: float
    demand_a: float
    demand_b: float
    total_demand: float
    allocated_a: float
    allocated_b: float
    total_allocated: float
    remaining_capacity: float
    utilization_pct: float
    is_throttled: bool
    policy_description: str


@dataclass
class CellAnalysis:
    """Stores payoff, best response indicators, and equilibrium status for a matrix cell."""
    strategy_a: str
    strategy_b: str
    payoff_a: float
    payoff_b: float
    is_a_best_response: bool
    is_b_best_response: bool
    is_nash_equilibrium: bool
    social_welfare: float


def compute_allocation(
    capacity: float,
    demand_a: float,
    demand_b: float,
) -> AllocationResult:
    """
    Computes CPU allocation enforcing capacity constraints.
    
    Allocation Policy:
    - If total requested demand <= capacity: each application receives its full request.
    - If total requested demand > capacity: proportional scaling is applied:
        allocated_i = capacity * (demand_i / total_demand)
    - If capacity <= 0 or total_demand == 0: allocation is 0.
    """
    capacity = max(0.0, float(capacity))
    demand_a = max(0.0, float(demand_a))
    demand_b = max(0.0, float(demand_b))
    total_demand = demand_a + demand_b

    if capacity <= 0.0 or total_demand == 0.0:
        allocated_a = 0.0
        allocated_b = 0.0
        total_allocated = 0.0
        is_throttled = total_demand > capacity
        policy_desc = (
            "Zero capacity available; no resources allocated."
            if capacity <= 0.0
            else "Zero demand requested; idle system."
        )
    elif total_demand <= capacity:
        allocated_a = demand_a
        allocated_b = demand_b
        total_allocated = total_demand
        is_throttled = False
        policy_desc = "Unconstrained Allocation: Total demand is within capacity limit."
    else:
        # Proportional scaling policy
        scaling_factor = capacity / total_demand
        allocated_a = demand_a * scaling_factor
        allocated_b = demand_b * scaling_factor
        total_allocated = capacity
        is_throttled = True
        policy_desc = (
            f"Proportional Fair Sharing: Demand ({total_demand:.1f}) exceeds capacity ({capacity:.1f}). "
            f"Allocations scaled to {scaling_factor * 100:.1f}% of requested units."
        )

    remaining_capacity = max(0.0, capacity - total_allocated)
    utilization_pct = (total_allocated / capacity * 100.0) if capacity > 0.0 else 0.0

    return AllocationResult(
        capacity=capacity,
        demand_a=round(demand_a, 2),
        demand_b=round(demand_b, 2),
        total_demand=round(total_demand, 2),
        allocated_a=round(allocated_a, 2),
        allocated_b=round(allocated_b, 2),
        total_allocated=round(total_allocated, 2),
        remaining_capacity=round(remaining_capacity, 2),
        utilization_pct=round(utilization_pct, 2),
        is_throttled=is_throttled,
        policy_description=policy_desc,
    )


def compute_payoff(
    allocated: float,
    requested: float,
    total_demand: float,
    capacity: float,
    value_per_unit: float,
    cost_per_unit: float,
    congestion_penalty_coeff: float,
    shortfall_penalty_coeff: float,
) -> float:
    """
    Computes application utility / payoff from first principles:
    
      Payoff = Benefit(allocated) - Cost(requested) - Shortfall_Penalty - Congestion_Penalty
      
    Where:
      - Benefit(allocated) = value_per_unit * allocated
      - Cost(requested) = cost_per_unit * requested
      - Shortfall_Penalty = shortfall_penalty_coeff * max(0, requested - allocated)
      - Congestion_Penalty = congestion_penalty_coeff * max(0, total_demand - capacity)
      
    Payoff values represent dimensionless utility scores, not physical units.
    """
    benefit = value_per_unit * allocated
    cost = cost_per_unit * requested
    shortfall = max(0.0, requested - allocated)
    shortfall_penalty = shortfall_penalty_coeff * shortfall
    excess_demand = max(0.0, total_demand - capacity)
    congestion_penalty = congestion_penalty_coeff * excess_demand

    utility = benefit - cost - shortfall_penalty - congestion_penalty
    return round(utility, 2)


def generate_parametric_payoff_matrix(
    capacity: float,
    a_low_demand: float,
    a_high_demand: float,
    b_low_demand: float,
    b_high_demand: float,
    value_per_unit: float,
    cost_per_unit: float,
    congestion_penalty_coeff: float,
    shortfall_penalty_coeff: float,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Calculates the 2x2 payoff matrix for all strategy profiles:
      {(strategy_a, strategy_b): (payoff_a, payoff_b)}
    """
    demands_a = {STRATEGY_LOW: a_low_demand, STRATEGY_HIGH: a_high_demand}
    demands_b = {STRATEGY_LOW: b_low_demand, STRATEGY_HIGH: b_high_demand}

    matrix = {}
    for sa in STRATEGIES:
        for sb in STRATEGIES:
            da = demands_a[sa]
            db = demands_b[sb]
            alloc = compute_allocation(capacity, da, db)

            ua = compute_payoff(
                allocated=alloc.allocated_a,
                requested=da,
                total_demand=alloc.total_demand,
                capacity=capacity,
                value_per_unit=value_per_unit,
                cost_per_unit=cost_per_unit,
                congestion_penalty_coeff=congestion_penalty_coeff,
                shortfall_penalty_coeff=shortfall_penalty_coeff,
            )
            ub = compute_payoff(
                allocated=alloc.allocated_b,
                requested=db,
                total_demand=alloc.total_demand,
                capacity=capacity,
                value_per_unit=value_per_unit,
                cost_per_unit=cost_per_unit,
                congestion_penalty_coeff=congestion_penalty_coeff,
                shortfall_penalty_coeff=shortfall_penalty_coeff,
            )
            matrix[(sa, sb)] = (ua, ub)

    return matrix


def compute_best_responses(
    matrix: Dict[Tuple[str, str], Tuple[float, float]],
    tol: float = 1e-4,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Identifies pure-strategy best responses for both players.
    
    Handles ties: if multiple actions achieve the maximum payoff within tolerance `tol`,
    all such actions are classified as best responses.
    
    Returns:
      (best_responses_a, best_responses_b)
      where best_responses_a[sb] is the list of strategy choices for A that maximize A's payoff against sb,
      and best_responses_b[sa] is the list of strategy choices for B that maximize B's payoff against sa.
    """
    # Best responses for Player A against each strategy of B
    best_a: Dict[str, List[str]] = {}
    for sb in STRATEGIES:
        payoffs = {sa: matrix[(sa, sb)][0] for sa in STRATEGIES}
        max_payoff = max(payoffs.values())
        best_a[sb] = [
            sa for sa, p in payoffs.items() if math.isclose(p, max_payoff, abs_tol=tol)
        ]

    # Best responses for Player B against each strategy of A
    best_b: Dict[str, List[str]] = {}
    for sa in STRATEGIES:
        payoffs = {sb: matrix[(sa, sb)][1] for sb in STRATEGIES}
        max_payoff = max(payoffs.values())
        best_b[sa] = [
            sb for sb, p in payoffs.items() if math.isclose(p, max_payoff, abs_tol=tol)
        ]

    return best_a, best_b


def analyze_game(
    matrix: Dict[Tuple[str, str], Tuple[float, float]],
    tol: float = 1e-4,
) -> Tuple[Dict[Tuple[str, str], CellAnalysis], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Performs complete game-theoretic analysis:
    - Finds best responses for both players.
    - Identifies all pure-strategy Nash equilibria (cells where both players play best responses).
    - Identifies Pareto optimal / Social welfare maximizing profiles.
    
    Returns:
      - cells: mapping from (sa, sb) to CellAnalysis
      - equilibria: list of (sa, sb) that are Nash equilibria
      - social_optima: list of (sa, sb) that maximize total social welfare (U_a + U_b)
    """
    best_a, best_b = compute_best_responses(matrix, tol=tol)

    cells: Dict[Tuple[str, str], CellAnalysis] = {}
    equilibria: List[Tuple[str, str]] = []
    max_welfare = -float("inf")

    # Pass 1: compute cell properties and max social welfare
    for sa in STRATEGIES:
        for sb in STRATEGIES:
            ua, ub = matrix[(sa, sb)]
            is_a_br = sa in best_a[sb]
            is_b_br = sb in best_b[sa]
            is_ne = is_a_br and is_b_br
            welfare = round(ua + ub, 2)

            if is_ne:
                equilibria.append((sa, sb))

            if welfare > max_welfare:
                max_welfare = welfare

            cells[(sa, sb)] = CellAnalysis(
                strategy_a=sa,
                strategy_b=sb,
                payoff_a=ua,
                payoff_b=ub,
                is_a_best_response=is_a_br,
                is_b_best_response=is_b_br,
                is_nash_equilibrium=is_ne,
                social_welfare=welfare,
            )

    # Pass 2: identify social optima
    social_optima = [
        (sa, sb)
        for (sa, sb), cell in cells.items()
        if math.isclose(cell.social_welfare, max_welfare, abs_tol=tol)
    ]

    return cells, equilibria, social_optima


def validate_inputs(
    capacity: float,
    a_low: float,
    a_high: float,
    b_low: float,
    b_high: float,
) -> List[str]:
    """Validates simulation input values, returning a list of validation error strings."""
    errors = []
    if capacity <= 0:
        errors.append("Total CPU Capacity must be strictly greater than 0.")
    if a_low < 0 or a_high < 0 or b_low < 0 or b_high < 0:
        errors.append("Resource demand values cannot be negative.")
    if a_low > a_high:
        errors.append("Application A: Low Demand cannot be greater than High Demand.")
    if b_low > b_high:
        errors.append("Application B: Low Demand cannot be greater than High Demand.")
    return errors
