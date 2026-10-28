"""
test_game_theory.py
Automated test suite verifying mathematical correctness of game theory calculations,
pure-strategy Nash equilibrium detection, tie handling, and capacity enforcement.
"""

import unittest
from game_theory import (
    STRATEGY_LOW,
    STRATEGY_HIGH,
    compute_allocation,
    compute_payoff,
    generate_parametric_payoff_matrix,
    compute_best_responses,
    analyze_game,
    validate_inputs,
)


class TestResourceAllocation(unittest.TestCase):
    def test_unconstrained_allocation(self):
        """When total demand is less than or equal to capacity, full demand is granted."""
        alloc = compute_allocation(capacity=100, demand_a=30, demand_b=30)
        self.assertEqual(alloc.allocated_a, 30.0)
        self.assertEqual(alloc.allocated_b, 30.0)
        self.assertEqual(alloc.total_allocated, 60.0)
        self.assertEqual(alloc.remaining_capacity, 40.0)
        self.assertEqual(alloc.utilization_pct, 60.0)
        self.assertFalse(alloc.is_throttled)

    def test_proportional_fair_sharing_throttled(self):
        """When demand exceeds capacity, proportional allocation is strictly enforced."""
        alloc = compute_allocation(capacity=100, demand_a=70, demand_b=70)
        self.assertEqual(alloc.allocated_a, 50.0)
        self.assertEqual(alloc.allocated_b, 50.0)
        self.assertEqual(alloc.total_allocated, 100.0)
        self.assertEqual(alloc.remaining_capacity, 0.0)
        self.assertEqual(alloc.utilization_pct, 100.0)
        self.assertTrue(alloc.is_throttled)

    def test_asymmetric_throttling(self):
        """Asymmetric demands when sum > capacity scale proportionally."""
        # 40 + 80 = 120 > 100
        # A: 100 * (40/120) = 33.33
        # B: 100 * (80/120) = 66.67
        alloc = compute_allocation(capacity=100, demand_a=40, demand_b=80)
        self.assertAlmostEqual(alloc.allocated_a, 33.33, places=2)
        self.assertAlmostEqual(alloc.allocated_b, 66.67, places=2)
        self.assertAlmostEqual(alloc.total_allocated, 100.0, places=1)
        self.assertTrue(alloc.is_throttled)

    def test_zero_capacity(self):
        """Zero capacity does not crash and allocates zero."""
        alloc = compute_allocation(capacity=0, demand_a=50, demand_b=50)
        self.assertEqual(alloc.allocated_a, 0.0)
        self.assertEqual(alloc.allocated_b, 0.0)
        self.assertEqual(alloc.total_allocated, 0.0)
        self.assertEqual(alloc.utilization_pct, 0.0)

    def test_zero_demand(self):
        """Zero demand allocates zero with full remaining capacity."""
        alloc = compute_allocation(capacity=100, demand_a=0, demand_b=0)
        self.assertEqual(alloc.allocated_a, 0.0)
        self.assertEqual(alloc.allocated_b, 0.0)
        self.assertEqual(alloc.remaining_capacity, 100.0)
        self.assertEqual(alloc.utilization_pct, 0.0)


class TestNashEquilibrium(unittest.TestCase):
    def test_canonical_prisoners_dilemma_matrix(self):
        """
        Verify the canonical prompt matrix:
        | A / B | Low (B) | High (B) |
        | Low   | (5, 5)  | (2, 7)   |
        | High  | (7, 2)  | (3, 3)   |
        
        Unique PSNE must be (High Demand, High Demand).
        """
        matrix = {
            (STRATEGY_LOW, STRATEGY_LOW): (5.0, 5.0),
            (STRATEGY_LOW, STRATEGY_HIGH): (2.0, 7.0),
            (STRATEGY_HIGH, STRATEGY_LOW): (7.0, 2.0),
            (STRATEGY_HIGH, STRATEGY_HIGH): (3.0, 3.0),
        }

        best_a, best_b = compute_best_responses(matrix)
        self.assertEqual(best_a[STRATEGY_LOW], [STRATEGY_HIGH])
        self.assertEqual(best_a[STRATEGY_HIGH], [STRATEGY_HIGH])
        self.assertEqual(best_b[STRATEGY_LOW], [STRATEGY_HIGH])
        self.assertEqual(best_b[STRATEGY_HIGH], [STRATEGY_HIGH])

        cells, equilibria, social_optima = analyze_game(matrix)
        self.assertEqual(equilibria, [(STRATEGY_HIGH, STRATEGY_HIGH)])
        self.assertEqual(social_optima, [(STRATEGY_LOW, STRATEGY_LOW)])
        self.assertEqual(cells[(STRATEGY_HIGH, STRATEGY_HIGH)].is_nash_equilibrium, True)
        self.assertEqual(cells[(STRATEGY_LOW, STRATEGY_LOW)].is_nash_equilibrium, False)

    def test_coordination_game_multiple_equilibria(self):
        """
        Coordination game:
        (Low, Low) -> (6, 6)
        (Low, High) -> (1, 1)
        (High, Low) -> (1, 1)
        (High, High) -> (4, 4)
        
        Two pure-strategy Nash equilibria: (Low, Low) and (High, High).
        """
        matrix = {
            (STRATEGY_LOW, STRATEGY_LOW): (6.0, 6.0),
            (STRATEGY_LOW, STRATEGY_HIGH): (1.0, 1.0),
            (STRATEGY_HIGH, STRATEGY_LOW): (1.0, 1.0),
            (STRATEGY_HIGH, STRATEGY_HIGH): (4.0, 4.0),
        }

        cells, equilibria, _ = analyze_game(matrix)
        self.assertIn((STRATEGY_LOW, STRATEGY_LOW), equilibria)
        self.assertIn((STRATEGY_HIGH, STRATEGY_HIGH), equilibria)
        self.assertEqual(len(equilibria), 2)

    def test_matching_pennies_no_pure_equilibrium(self):
        """
        Cyclic game with no pure strategy equilibrium:
        (Low, Low) -> (1, -1)
        (Low, High) -> (-1, 1)
        (High, Low) -> (-1, 1)
        (High, High) -> (1, -1)
        """
        matrix = {
            (STRATEGY_LOW, STRATEGY_LOW): (1.0, -1.0),
            (STRATEGY_LOW, STRATEGY_HIGH): (-1.0, 1.0),
            (STRATEGY_HIGH, STRATEGY_LOW): (-1.0, 1.0),
            (STRATEGY_HIGH, STRATEGY_HIGH): (1.0, -1.0),
        }

        cells, equilibria, _ = analyze_game(matrix)
        self.assertEqual(len(equilibria), 0)

    def test_tie_handling_in_best_response(self):
        """When multiple strategies yield equal maximal payoff, all are best responses."""
        matrix = {
            (STRATEGY_LOW, STRATEGY_LOW): (4.0, 4.0),
            (STRATEGY_LOW, STRATEGY_HIGH): (4.0, 2.0),
            (STRATEGY_HIGH, STRATEGY_LOW): (4.0, 2.0),
            (STRATEGY_HIGH, STRATEGY_HIGH): (4.0, 4.0),
        }
        best_a, best_b = compute_best_responses(matrix)
        # Player A gets 4.0 in both choices against Low and against High
        self.assertEqual(len(best_a[STRATEGY_LOW]), 2)
        self.assertEqual(len(best_a[STRATEGY_HIGH]), 2)


class TestInputValidation(unittest.TestCase):
    def test_invalid_capacity(self):
        errors = validate_inputs(capacity=-10, a_low=20, a_high=50, b_low=20, b_high=50)
        self.assertTrue(any("Capacity must be strictly greater than 0" in e for e in errors))

    def test_negative_demands(self):
        errors = validate_inputs(capacity=100, a_low=-5, a_high=50, b_low=20, b_high=50)
        self.assertTrue(any("cannot be negative" in e for e in errors))

    def test_low_greater_than_high(self):
        errors = validate_inputs(capacity=100, a_low=80, a_high=30, b_low=20, b_high=50)
        self.assertTrue(any("cannot be greater than High" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
