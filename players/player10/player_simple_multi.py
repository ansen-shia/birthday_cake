#!/usr/bin/env python3
"""
Simple multi-stage player that just modifies the tolerance and collects multiple solutions.
"""

import sys
import os
import random
from typing import List, Tuple, Optional
from shapely.geometry import Point
from shapely.ops import split
from statistics import stdev

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from players.player10.player import Player10


class Player10SimpleMulti(Player10):
    """Simple multi-stage player that uses 0.25 cm² tolerance and collects multiple solutions."""

    def __init__(
        self,
        children: int,
        cake,
        cake_path: str,
        num_angle_attempts: int = 20,
        max_solutions: int = 5,
    ) -> None:
        super().__init__(children, cake, cake_path)

        # Use 0.25 cm² tolerance as requested
        self.target_area_tolerance = 0.25

        # Number of different angles to try
        self.num_angle_attempts = num_angle_attempts
        # Maximum number of solutions to collect
        self.max_solutions = max_solutions

    def evaluate_solution(
        self, cuts: List[Tuple[Point, Point]], cake_copy
    ) -> Tuple[float, float, float, List[float], List[float]]:
        """Evaluate a complete solution and return (area_span, area_std, ratio_variance, areas, ratios)."""
        # Apply all cuts to a copy
        test_cake = cake_copy.copy()

        for from_p, to_p in cuts:
            try:
                test_cake.cut(from_p, to_p)
            except Exception:
                return float("inf"), float("inf"), float("inf"), [], []

        # Check if we got the right number of pieces
        pieces = test_cake.get_pieces()
        if len(pieces) != self.children:
            return float("inf"), float("inf"), float("inf"), [], []

        # Calculate areas and ratios
        areas = [p.area for p in pieces]
        ratios = test_cake.get_piece_ratios()

        # Calculate area span (max - min)
        area_span = max(areas) - min(areas)

        # Calculate area standard deviation
        if len(areas) > 1:
            area_std = stdev(areas)
        else:
            area_std = 0.0

        # Calculate ratio variance
        if len(ratios) > 1:
            ratio_variance = stdev(ratios)
        else:
            ratio_variance = 0.0

        return area_span, area_std, ratio_variance, areas, ratios

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic - collect multiple solutions and pick best crust ratio."""
        target_area = self.cake.get_area() / self.children
        target_ratio = (
            self.cake.get_piece_ratios()[0] if self.cake.get_pieces() else 0.5
        )

        print(
            f"__________Simple Multi-Stage Cutting for {self.children} children_______"
        )
        print(f"TARGET AREA: {target_area:.2f} cm²")
        print(f"TARGET CRUST RATIO: {target_ratio:.3f}")
        print(f"TOLERANCE: {self.target_area_tolerance} cm²")
        print(
            "Strategy: Collect multiple solutions with 0.25 cm² tolerance, optimize crust ratios"
        )

        # Collect multiple solutions
        solutions = []
        attempts = 0
        max_attempts = self.num_angle_attempts * 2

        print("=== Collecting solutions with 0.25 cm² tolerance ===")

        while len(solutions) < self.max_solutions and attempts < max_attempts:
            attempts += 1

            # Generate a single solution using the original approach
            solution = self._generate_single_solution(target_area, target_ratio)

            if solution is None:
                continue

            cuts, area_span, area_std, ratio_variance, areas, ratios = solution

            # Check if this solution meets our tolerance (area span < 0.5 cm²)
            if area_span <= 0.5:
                solutions.append(
                    (cuts, area_span, area_std, ratio_variance, areas, ratios)
                )
                print(
                    f"  Solution {len(solutions)}: area_span={area_span:.3f}, ratio_var={ratio_variance:.4f}"
                )

        print(f"Collected {len(solutions)} valid solutions out of {attempts} attempts")

        if not solutions:
            print("No valid solutions found! Using single solution approach...")
            return self._fallback_single_solution(target_area, target_ratio)

        # Pick the solution with the best crust ratio variance
        best_solution = min(solutions, key=lambda x: x[3])  # x[3] is ratio_variance

        (
            best_cuts,
            best_area_span,
            best_area_std,
            best_ratio_variance,
            best_areas,
            best_ratios,
        ) = best_solution

        print("\nBest solution selected:")
        print(f"  Area span: {best_area_span:.3f} cm²")
        print(f"  Area std dev: {best_area_std:.3f} cm²")
        print(f"  Ratio variance: {best_ratio_variance:.4f}")
        print(f"  Areas: {[f'{a:.2f}' for a in sorted(best_areas)]}")
        print(f"  Ratios: {[f'{r:.3f}' for r in best_ratios]}")

        # Final validation
        spec_threshold = 0.5
        print("\nFinal validation:")
        print(f"  Area span: {best_area_span:.3f} cm²")
        print(f"  Specification threshold: {spec_threshold} cm²")
        if best_area_span < spec_threshold:
            print("  ✓ SPECIFICATION COMPLIANT")
        else:
            print("  ⚠️  SPECIFICATION VIOLATION")

        return best_cuts

    def _generate_single_solution(
        self, target_area: float, target_ratio: float
    ) -> Optional[
        Tuple[List[Tuple[Point, Point]], float, float, float, List[float], List[float]]
    ]:
        """Generate a single solution using the original approach but with 0.25 cm² tolerance."""
        cake_copy = self.cake.copy()
        all_cuts = []

        # Initialize: the whole cake is for all children
        pieces_queue = [(cake_copy.exterior_shape, self.children)]

        cut_number = 0
        while cut_number < self.children - 1:
            # Find a piece that needs to be divided
            cutting_piece = None
            cutting_num_children = 0
            cutting_index = -1

            for i, (piece, num_children) in enumerate(pieces_queue):
                if num_children > 1:
                    if num_children > cutting_num_children:
                        cutting_piece = piece
                        cutting_num_children = num_children
                        cutting_index = i
                    elif num_children == cutting_num_children and piece.area > (
                        cutting_piece.area if cutting_piece else 0
                    ):
                        cutting_piece = piece
                        cutting_num_children = num_children
                        cutting_index = i

            if cutting_piece is None:
                break

            # Remove the piece from queue
            pieces_queue.pop(cutting_index)

            # Try different split ratios
            min_split = 1
            max_split = max(1, cutting_num_children // 2)
            split_children = random.randint(min_split, max_split)

            remaining_children = cutting_num_children - split_children
            target_cut_area = target_area * split_children

            # Try different angles
            angle = random.uniform(0, 180)

            # Find the cut position using binary search (with 0.25 cm² tolerance)
            position = self.binary_search(cutting_piece, target_cut_area, angle)
            if position is None:
                return None  # Failed to find valid cut

            cut_line = self.find_line(position, cutting_piece, angle)
            cut_points = self.find_cuts(cut_line, cutting_piece)
            if cut_points is None:
                return None  # Failed to find valid cut points

            from_p, to_p = cut_points

            # Simulate the cut to get the two pieces
            test_pieces = split(cutting_piece, cut_line)
            if len(test_pieces.geoms) != 2:
                return None  # Invalid cut

            p1, p2 = test_pieces.geoms

            # Determine which piece is for split_children
            if abs(p1.area - target_cut_area) < abs(p2.area - target_cut_area):
                small_piece, large_piece = p1, p2
            else:
                small_piece, large_piece = p2, p1

            # Make the cut on the actual cake
            try:
                cake_copy.cut(from_p, to_p)
                all_cuts.append((from_p, to_p))
                cut_number += 1

                # Add the two new pieces to the queue
                pieces_queue.append((small_piece, split_children))
                pieces_queue.append((large_piece, remaining_children))
            except Exception:
                return None

        # Evaluate the complete solution
        area_span, area_std, ratio_variance, areas, ratios = self.evaluate_solution(
            all_cuts, self.cake
        )
        return (all_cuts, area_span, area_std, ratio_variance, areas, ratios)

    def _fallback_single_solution(
        self, target_area: float, target_ratio: float
    ) -> List[Tuple[Point, Point]]:
        """Fallback to the original single-solution approach."""
        print("Using fallback single-solution approach...")
        return super().get_cuts()
