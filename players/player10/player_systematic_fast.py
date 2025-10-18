#!/usr/bin/env python3
"""
Fast systematic multi-stage algorithm.
Uses 0.25 cm² tolerance and collects multiple solutions efficiently.
"""

import sys
import os
import random
from typing import List, Tuple, Optional
from shapely.geometry import Point, Polygon
from shapely.ops import split
from statistics import stdev

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from players.player10.player import Player10


class Player10SystematicFast(Player10):
    """Fast systematic multi-stage player that collects multiple solutions with 0.25 cm² tolerance."""

    def __init__(
        self,
        children: int,
        cake,
        cake_path: str,
        num_angle_attempts: int = 20,
        area_std_threshold: float = 0.5,
        max_solutions: int = 10,
    ) -> None:
        super().__init__(children, cake, cake_path)

        # Use fixed tight tolerance to ensure no piece is more than 0.25 cm² from target
        self.target_area_tolerance = 0.25

        # Number of different angles to try
        self.num_angle_attempts = num_angle_attempts
        # Maximum number of valid solutions to collect
        self.max_solutions = max_solutions
        # Area span threshold for filtering
        self.area_std_threshold = area_std_threshold

    def binary_search_all_solutions(
        self, piece: Polygon, target_area: float, angle: float
    ) -> List[float]:
        """Binary search that collects ALL positions within 0.25 cm² tolerance."""
        left_pos = 0.0
        right_pos = 1.0
        valid_positions = []

        # Try for best cut for 50 iterations
        for iteration in range(50):
            mid_pos = (left_pos + right_pos) / 2
            cut_area = self.calculate_piece_area(piece, mid_pos, angle)

            if cut_area == 0:
                left_pos = mid_pos
                continue

            if cut_area >= piece.area:
                right_pos = mid_pos
                continue

            error = abs(cut_area - target_area)

            # If within tolerance, add to valid positions
            if error < self.target_area_tolerance:
                valid_positions.append(mid_pos)

            # Adjust search
            if cut_area < target_area:
                left_pos = mid_pos
            else:
                right_pos = mid_pos

        return valid_positions

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

    def generate_multiple_solutions(
        self, target_area: float, target_ratio: float
    ) -> List[
        Tuple[List[Tuple[Point, Point]], float, float, float, List[float], List[float]]
    ]:
        """Generate multiple solutions using systematic approach with 0.25 cm² tolerance."""
        print("=== Generating solutions with 0.25 cm² tolerance ===")

        all_solutions = []
        attempts = 0
        max_attempts = self.num_angle_attempts * 3  # More attempts

        while len(all_solutions) < self.max_solutions and attempts < max_attempts:
            attempts += 1

            # Generate a single solution using the original approach but with tighter tolerance
            solution = self._generate_single_solution_systematic(
                target_area, target_ratio
            )

            if solution is None:
                continue

            cuts, area_span, area_std, ratio_variance, areas, ratios = solution

            # Check if this solution meets our area span threshold
            if area_span <= self.area_std_threshold:
                all_solutions.append(
                    (cuts, area_span, area_std, ratio_variance, areas, ratios)
                )
                print(
                    f"  Solution {len(all_solutions)}: area_span={area_span:.3f}, ratio_var={ratio_variance:.4f}"
                )

        print(
            f"Generated {len(all_solutions)} valid solutions out of {attempts} attempts"
        )
        return all_solutions

    def _generate_single_solution_systematic(
        self, target_area: float, target_ratio: float
    ) -> Optional[
        Tuple[List[Tuple[Point, Point]], float, float, float, List[float], List[float]]
    ]:
        """Generate a single solution using systematic approach with 0.25 cm² tolerance."""
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

            # Find ALL valid positions for this angle
            valid_positions = self.binary_search_all_solutions(
                cutting_piece, target_cut_area, angle
            )

            if not valid_positions:
                return None  # No valid positions found

            # Choose a random valid position
            position = random.choice(valid_positions)

            cut_line = self.find_line(position, cutting_piece, angle)
            cut_points = self.find_cuts(cut_line, cutting_piece)
            if cut_points is None:
                return None

            from_p, to_p = cut_points

            # Simulate the cut to get the two pieces
            test_pieces = split(cutting_piece, cut_line)
            if len(test_pieces.geoms) != 2:
                return None

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

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic - systematic approach with 0.25 cm² tolerance."""
        target_area = self.cake.get_area() / self.children
        target_ratio = (
            self.cake.get_piece_ratios()[0] if self.cake.get_pieces() else 0.5
        )

        print("__________Systematic Fast Cutting for {self.children} children_______")
        print("TARGET AREA: {target_area:.2f} cm²")
        print("TARGET CRUST RATIO: {target_ratio:.3f}")
        print("TOLERANCE: {self.target_area_tolerance} cm²")
        print("Strategy: Systematic solution collection with 0.25 cm² tolerance")

        # Stage 1: Generate multiple solutions
        all_solutions = self.generate_multiple_solutions(target_area, target_ratio)

        if not all_solutions:
            print(
                "No valid solutions found! Falling back to single solution approach..."
            )
            return self._fallback_single_solution(target_area, target_ratio)

        # Stage 2: Optimize for best crust ratio among valid solutions
        best_solution = min(all_solutions, key=lambda x: x[3])  # x[3] is ratio_variance

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
        print("  Area span: {best_area_span:.3f} cm²")
        print("  Specification threshold: {spec_threshold} cm²")
        if best_area_span < spec_threshold:
            print("  ✓ SPECIFICATION COMPLIANT")
        else:
            print("  ⚠️  SPECIFICATION VIOLATION")

        return best_cuts

    def _fallback_single_solution(
        self, target_area: float, target_ratio: float
    ) -> List[Tuple[Point, Point]]:
        """Fallback to the original single-solution approach."""
        print("Using fallback single-solution approach...")
        return super().get_cuts()
