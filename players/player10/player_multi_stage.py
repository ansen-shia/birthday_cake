from shapely.geometry import LineString, Point, Polygon
import math
import random
from statistics import stdev
from typing import List, Tuple, Optional

from players.player import Player
from src.cake import Cake
from shapely.ops import split

NUMBER_ATTEMPTS = 360


class Player10MultiStage(Player):
    def __init__(
        self,
        children: int,
        cake: Cake,
        cake_path: str | None,
        num_angle_attempts: int = NUMBER_ATTEMPTS,
        area_std_threshold: float = 0.5,  # cm² - threshold for area standard deviation
        max_solutions: int = 10,  # Maximum number of valid solutions to collect
    ) -> None:
        super().__init__(children, cake, cake_path)

        # Calculate adaptive tolerances based on number of children and target area
        self.target_area = cake.get_area() / children

        # Use fixed tight tolerance to ensure no piece is more than 0.25 cm² from target
        # This ensures area span requirements are met
        self.target_area_tolerance = 0.25

        # Adaptive area std dev threshold: also scales with children
        self.area_std_threshold = area_std_threshold

        # Number of different angles to try (more attempts = better for complex shapes)
        self.num_angle_attempts = num_angle_attempts
        # Maximum number of valid solutions to collect in Stage 1
        self.max_solutions = max_solutions

        print(f"Adaptive tolerances for {children} children:")
        print(f"  Target area per child: {self.target_area:.2f} cm²")
        print(f"  Binary search tolerance: {self.target_area_tolerance:.3f} cm²")
        print(f"  Area std dev threshold: {self.area_std_threshold:.3f} cm²")

    def find_line(self, position: float, piece: Polygon, angle: float):
        """Make a line at a given angle through a position that cuts the piece.

        Args:
            position: Position along the sweep direction (0 to 1)
            piece: The polygon piece to cut
            angle: Angle in degrees (0-360) where 0 is right, 90 is up
        """

        # Get bounding box of piece
        leftmost, lowest, rightmost, highest = piece.bounds
        width = rightmost - leftmost
        height = highest - lowest
        max_dim = max(width, height) * 2

        # Convert angle to radians
        angle_rad = math.radians(angle)

        # Calculate the perpendicular direction for the sweep
        sweep_angle = angle_rad + math.pi / 2

        # Start from center of bounding box
        center_x = (leftmost + rightmost) / 2
        center_y = (lowest + highest) / 2

        # Calculate sweep offset based on position (0 to 1)
        sweep_offset = (position - 0.5) * max_dim
        offset_x = sweep_offset * math.cos(sweep_angle)
        offset_y = sweep_offset * math.sin(sweep_angle)

        # Calculate point on the sweep line
        point_x = center_x + offset_x
        point_y = center_y + offset_y

        # Create a line at the given angle through this point
        dx = math.cos(angle_rad) * max_dim
        dy = math.sin(angle_rad) * max_dim

        # Create line extending in both directions
        start_point = (point_x - dx, point_y - dy)
        end_point = (point_x + dx, point_y + dy)
        cut_line = LineString([start_point, end_point])

        return cut_line

    def find_cuts(self, line: LineString, piece: Polygon):
        """Find exactly two points where the cut line intersects the cake boundary, ensuring only one cut per turn."""
        intersection = line.intersection(piece.boundary)

        # Collect all intersection points
        points = []
        if intersection.is_empty:
            return None  # No intersection
        if intersection.geom_type == "Point":
            points = [intersection]
        elif intersection.geom_type == "MultiPoint":
            points = list(intersection.geoms)
        elif intersection.geom_type == "LineString":
            coords = list(intersection.coords)
            points = [Point(coords[0]), Point(coords[-1])]
        elif intersection.geom_type == "GeometryCollection":
            for geom in intersection.geoms:
                if geom.geom_type == "Point":
                    points.append(geom)
                elif geom.geom_type == "LineString":
                    coords = list(geom.coords)
                    points.extend([Point(coords[0]), Point(coords[-1])])

        unique_points = []
        tolerance = 1e-6
        for p in points:
            is_duplicate = False
            for q in unique_points:
                if p.distance(q) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(p)
        points = unique_points

        if len(points) < 2:
            return None  # Not enough points for a valid cut

        # If exactly 2 points, use them
        if len(points) == 2:
            return (points[0], points[1])

        # If more than 2 points, we need to find the pair that creates a valid cut
        # A valid cut should split the piece into exactly 2 pieces
        # Try all pairs and find the one that works
        from itertools import combinations

        for p1, p2 in combinations(points, 2):
            test_line = LineString([p1, p2])
            # Check if this line segment is mostly inside the piece
            # by checking if the midpoint is inside
            midpoint = test_line.interpolate(0.5, normalized=True)
            if piece.contains(midpoint) or piece.boundary.contains(midpoint):
                # Also verify this cut would split into exactly 2 pieces
                from shapely.ops import split as shapely_split

                result = shapely_split(piece, test_line)
                if len(result.geoms) == 2:
                    return (p1, p2)

        # Fallback: if no valid pair found, return None
        return None

    def calculate_piece_area(self, piece: Polygon, position: float, angle: float):
        """Determines the area of the pieces we cut.

        Args:
            piece: The polygon piece to cut
            position: Position along sweep direction (0 to 1)
            angle: Angle in degrees for the cutting line
        """
        line = self.find_line(position, piece, angle)
        pieces = split(piece, line)

        # we should get two pieces if not, line didn't cut properly
        if len(pieces.geoms) != 2:
            # if we're at the extremes
            if position <= 0.0:
                return 0.0
            elif position >= 1.0:
                return piece.area
            else:
                return 0.0

        piece1, piece2 = pieces.geoms

        # Calculate which piece is "first" based on sweep direction
        angle_rad = math.radians(angle)
        sweep_angle = angle_rad + math.pi / 2
        sweep_dir_x = math.cos(sweep_angle)
        sweep_dir_y = math.sin(sweep_angle)

        # Project centroids onto sweep direction
        centroid1 = piece1.centroid
        centroid2 = piece2.centroid

        proj1 = centroid1.x * sweep_dir_x + centroid1.y * sweep_dir_y
        proj2 = centroid2.x * sweep_dir_x + centroid2.y * sweep_dir_y

        # Return the area of the "first" piece in sweep direction
        if proj1 < proj2:
            return piece1.area
        else:
            return piece2.area

    def binary_search(self, piece: Polygon, target_area: float, angle: float):
        """Use binary search to find position along sweep that cuts off target_area.

        Args:
            piece: The polygon piece to cut
            target_area: The target area for the cut piece
            angle: Angle in degrees for the cutting line
        """

        left_pos = 0.0
        right_pos = 1.0
        best_pos = None
        best_error = float("inf")

        # try for best cut for 50 iterations
        for iteration in range(50):
            # try middle first
            mid_pos = (left_pos + right_pos) / 2

            # get the area of that prospective position
            cut_area = self.calculate_piece_area(piece, mid_pos, angle)

            if cut_area == 0:
                # Too far left, move right
                left_pos = mid_pos
                continue

            if cut_area >= piece.area:
                # Too far right, move left
                right_pos = mid_pos
                continue

            # how far away from the target value
            error = abs(cut_area - target_area)

            # Track best
            if error < best_error:
                best_error = error
                best_pos = mid_pos

            # Check if it's good enough
            if error < self.target_area_tolerance:
                return mid_pos

            # Adjust search based on distance from target area
            if cut_area < target_area:
                left_pos = mid_pos  # Need more, move right
            else:
                right_pos = mid_pos  # Too much, move left

        return best_pos

    def binary_search_all_solutions(
        self, piece: Polygon, target_area: float, angle: float
    ) -> List[float]:
        """Binary search that collects ALL positions within tolerance instead of just the best one.

        Args:
            piece: The polygon piece to cut
            target_area: The target area for the cut piece
            angle: Angle in degrees for the cutting line

        Returns:
            List of all valid positions within tolerance
        """
        left_pos = 0.0
        right_pos = 1.0
        valid_positions = []

        # Try for best cut for 50 iterations
        for iteration in range(50):
            # Try middle first
            mid_pos = (left_pos + right_pos) / 2

            # Get the area of that prospective position
            cut_area = self.calculate_piece_area(piece, mid_pos, angle)

            if cut_area == 0:
                # Too far left, move right
                left_pos = mid_pos
                continue

            if cut_area >= piece.area:
                # Too far right, move left
                right_pos = mid_pos
                continue

            # How far away from the target value
            error = abs(cut_area - target_area)

            # If within tolerance, add to valid positions
            if error < self.target_area_tolerance:
                valid_positions.append(mid_pos)

            # Adjust search based on distance from target area
            if cut_area < target_area:
                left_pos = mid_pos  # Need more, move right
            else:
                right_pos = mid_pos  # Too much, move left

        return valid_positions

    def evaluate_solution(
        self, cuts: List[Tuple[Point, Point]], cake_copy: Cake
    ) -> Tuple[float, float, float, List[float], List[float]]:
        """Evaluate a complete solution and return (area_span, area_std, ratio_variance, areas, ratios).

        Args:
            cuts: List of cuts to apply
            cake_copy: Cake to test on (will be modified)

        Returns:
            Tuple of (area_span, area_std_dev, ratio_variance, areas, ratios)
        """
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

        # Calculate area span (max - min) - this is what the specification uses
        area_span = max(areas) - min(areas)

        # Calculate area standard deviation (for comparison)
        if len(areas) > 1:
            area_std = stdev(areas)
        else:
            area_std = 0.0

        # Calculate ratio variance (crust ratio variance)
        if len(ratios) > 1:
            ratio_variance = stdev(ratios)
        else:
            ratio_variance = 0.0

        return area_span, area_std, ratio_variance, areas, ratios

    def stage1_find_valid_solutions(
        self, target_area: float, target_ratio: float
    ) -> List[
        Tuple[List[Tuple[Point, Point]], float, float, float, List[float], List[float]]
    ]:
        """Stage 1: Systematically find ALL valid solutions using binary search with 0.25 cm² tolerance.

        Returns:
            List of (cuts, area_span, area_std, ratio_variance, areas, ratios) tuples
        """
        print("=== STAGE 1: Systematic solution collection with 0.25 cm² tolerance ===")

        # Use systematic approach to collect all possible solutions
        all_solutions = self._generate_all_systematic_solutions(
            target_area, target_ratio
        )

        # Filter solutions that meet area span threshold
        valid_solutions = []
        for solution in all_solutions:
            cuts, area_span, area_std, ratio_variance, areas, ratios = solution

            # Check if this solution meets our area span threshold
            if area_span <= self.area_std_threshold:
                valid_solutions.append(
                    (cuts, area_span, area_std, ratio_variance, areas, ratios)
                )
                print(
                    f"  Solution {len(valid_solutions)}: area_span={area_span:.3f}, area_std={area_std:.3f}, ratio_var={ratio_variance:.4f}"
                )

                # Show piece areas for this solution
                print(f"    Areas: {[f'{a:.2f}' for a in sorted(areas)]}")
                print(f"    Ratios: {[f'{r:.3f}' for r in ratios]}")

        print(
            f"Stage 1 complete: Found {len(valid_solutions)} valid solutions out of {len(all_solutions)} total solutions"
        )
        return valid_solutions

    def _generate_all_systematic_solutions(
        self, target_area: float, target_ratio: float
    ) -> List[
        Tuple[List[Tuple[Point, Point]], float, float, float, List[float], List[float]]
    ]:
        """Generate ALL possible solutions using systematic binary search with 0.25 cm² tolerance.

        This method explores all possible cut combinations by:
        1. For each cut, finding ALL valid positions within 0.25 cm² tolerance
        2. Systematically exploring combinations of these positions
        3. Collecting all complete solutions that meet the tolerance
        """
        print("  Exploring all systematic solutions with 0.25 cm² tolerance...")

        all_solutions = []
        cake_copy = self.cake.copy()

        # Initialize: the whole cake is for all children
        pieces_queue = [(cake_copy.exterior_shape, self.children)]

        # Recursively explore all possible cut combinations
        self._explore_cut_combinations(pieces_queue, [], target_area, all_solutions, 0)

        print(f"  Generated {len(all_solutions)} total solutions")
        return all_solutions

    def _explore_cut_combinations(
        self,
        pieces_queue: List[Tuple[Polygon, int]],
        current_cuts: List[Tuple[Point, Point]],
        target_area: float,
        all_solutions: List,
        cut_number: int,
    ):
        """Recursively explore all possible cut combinations."""

        # Base case: if we have enough pieces for all children
        if len(pieces_queue) == self.children:
            # Evaluate this complete solution
            area_span, area_std, ratio_variance, areas, ratios = self.evaluate_solution(
                current_cuts, self.cake
            )
            all_solutions.append(
                (
                    current_cuts.copy(),
                    area_span,
                    area_std,
                    ratio_variance,
                    areas,
                    ratios,
                )
            )
            return

        # Find a piece that needs to be divided (num_children > 1)
        cutting_piece = None
        cutting_num_children = 0
        cutting_index = -1

        for i, (piece, num_children) in enumerate(pieces_queue):
            if num_children > 1:
                # Prefer larger pieces or pieces with more children
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
            return  # No more pieces to cut

        # Remove the piece from queue
        new_pieces_queue = pieces_queue.copy()
        new_pieces_queue.pop(cutting_index)

        # Try different split ratios: split n children into (k, n-k)
        min_split = 1
        max_split = max(1, cutting_num_children // 2)

        for split_children in range(min_split, max_split + 1):
            remaining_children = cutting_num_children - split_children
            target_cut_area = target_area * split_children

            # Try different angles
            for angle in range(0, 180, 10):  # Try every 10 degrees
                # Find ALL valid positions for this angle
                valid_positions = self.binary_search_all_solutions(
                    cutting_piece, target_cut_area, angle
                )

                # Try each valid position
                for position in valid_positions:
                    cut_line = self.find_line(position, cutting_piece, angle)
                    cut_points = self.find_cuts(cut_line, cutting_piece)
                    if cut_points is None:
                        continue

                    from_p, to_p = cut_points

                    # Simulate the cut to get the two pieces
                    test_pieces = split(cutting_piece, cut_line)
                    if len(test_pieces.geoms) != 2:
                        continue

                    p1, p2 = test_pieces.geoms

                    # Determine which piece is for split_children
                    if abs(p1.area - target_cut_area) < abs(p2.area - target_cut_area):
                        small_piece, large_piece = p1, p2
                    else:
                        small_piece, large_piece = p2, p1

                    # Add the cut to current cuts
                    new_cuts = current_cuts + [(from_p, to_p)]

                    # Add the two new pieces to the queue
                    new_queue = new_pieces_queue + [
                        (small_piece, split_children),
                        (large_piece, remaining_children),
                    ]

                    # Recursively explore this branch
                    self._explore_cut_combinations(
                        new_queue, new_cuts, target_area, all_solutions, cut_number + 1
                    )

    def _generate_single_solution(
        self, target_area: float, target_ratio: float
    ) -> Optional[
        Tuple[List[Tuple[Point, Point]], float, float, float, List[float], List[float]]
    ]:
        """Generate a single complete solution using the divide-and-conquer approach."""
        cake_copy = self.cake.copy()
        all_cuts = []

        # Initialize: the whole cake is for all children
        pieces_queue = [(cake_copy.exterior_shape, self.children)]

        cut_number = 0
        while cut_number < self.children - 1:
            # Find a piece that needs to be divided (num_children > 1)
            cutting_piece = None
            cutting_num_children = 0
            cutting_index = -1

            for i, (piece, num_children) in enumerate(pieces_queue):
                if num_children > 1:
                    # Prefer larger pieces or pieces with more children
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
                # All pieces are for 1 child
                break

            # Remove the piece from queue
            pieces_queue.pop(cutting_index)

            # Try different split ratios: split n children into (k, n-k)
            min_split = 1
            max_split = max(1, cutting_num_children // 2)

            # Try a random split ratio and angle
            split_children = random.randint(min_split, max_split)
            angle = random.uniform(0, 180)

            remaining_children = cutting_num_children - split_children
            target_cut_area = target_area * split_children

            # Find the cut position using binary search
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

                # Add the two new pieces to the queue with their child counts
                pieces_queue.append((small_piece, split_children))
                pieces_queue.append((large_piece, remaining_children))
            except Exception:
                return None  # Cut failed

        # Evaluate the complete solution
        area_span, area_std, ratio_variance, areas, ratios = self.evaluate_solution(
            all_cuts, self.cake
        )
        return (all_cuts, area_span, area_std, ratio_variance, areas, ratios)

    def stage2_optimize_crust_ratios(
        self,
        valid_solutions: List[
            Tuple[
                List[Tuple[Point, Point]], float, float, float, List[float], List[float]
            ]
        ],
    ) -> List[Tuple[Point, Point]]:
        """Stage 2: Among valid solutions, find the one with the best crust ratio distribution.

        Args:
            valid_solutions: List of valid solutions from Stage 1

        Returns:
            The best solution's cuts
        """
        if not valid_solutions:
            print("No valid solutions found in Stage 1!")
            return []

        print(
            f"=== STAGE 2: Optimizing crust ratios among {len(valid_solutions)} valid solutions ==="
        )

        # HARD THRESHOLD: Filter solutions that meet specification requirement
        # Specification: area span < 0.5 cm² (this is what the spec actually uses)
        spec_threshold = 0.5
        spec_compliant_solutions = [
            sol
            for sol in valid_solutions
            if sol[1] < spec_threshold  # sol[1] is area_span
        ]

        if not spec_compliant_solutions:
            print(
                f"⚠️  WARNING: No solutions meet specification threshold ({spec_threshold} cm²)"
            )
            print(
                f"   Best solution has area span: {min(sol[1] for sol in valid_solutions):.3f} cm²"
            )
            print("   Using best available solution despite specification violation...")
            # Use the best solution even if it doesn't meet spec
            solutions_to_consider = valid_solutions
        else:
            print(
                f"✓ {len(spec_compliant_solutions)}/{len(valid_solutions)} solutions meet specification threshold ({spec_threshold} cm²)"
            )
            solutions_to_consider = spec_compliant_solutions

        # Sort by ratio variance (lower is better)
        solutions_to_consider.sort(key=lambda x: x[3])  # x[3] is ratio_variance

        best_solution = solutions_to_consider[0]
        (
            best_cuts,
            best_area_span,
            best_area_std,
            best_ratio_variance,
            best_areas,
            best_ratios,
        ) = best_solution

        print("Best solution selected:")
        print(
            f"  Area span: {best_area_span:.3f} cm² {'✓' if best_area_span < spec_threshold else '⚠️'}"
        )
        print(f"  Area std dev: {best_area_std:.3f} cm²")
        print(f"  Ratio variance: {best_ratio_variance:.4f}")
        print(f"  Areas: {[f'{a:.2f}' for a in sorted(best_areas)]}")
        print(f"  Ratios: {[f'{r:.3f}' for r in best_ratios]}")

        return best_cuts

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic - two-stage optimization approach"""
        print(f"__________Multi-Stage Cutting for {self.children} children_______")

        target_area = self.cake.get_area() / self.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area
        print(f"TARGET AREA: {target_area:.2f} cm²")
        print(f"TARGET CRUST RATIO: {target_ratio:.3f}")
        print(f"ADAPTIVE AREA STD DEV THRESHOLD: {self.area_std_threshold:.3f} cm²")
        print(
            "Strategy: Two-stage optimization (area threshold → crust optimization)\n"
        )

        # Stage 1: Find multiple valid solutions meeting area threshold
        valid_solutions = self.stage1_find_valid_solutions(target_area, target_ratio)

        if not valid_solutions:
            print(
                "No valid solutions found! Falling back to single solution approach..."
            )
            # Fallback to original approach
            return self._fallback_single_solution(target_area, target_ratio)

        # Stage 2: Optimize for best crust ratio among valid solutions
        best_cuts = self.stage2_optimize_crust_ratios(valid_solutions)

        # Final validation: Check if result meets specification
        if best_cuts:
            (
                final_area_span,
                final_area_std,
                final_ratio_variance,
                final_areas,
                final_ratios,
            ) = self.evaluate_solution(best_cuts, self.cake)
            spec_threshold = 0.5

            print(f"\n{'=' * 50}")
            print("FINAL VALIDATION:")
            print(f"  Area span: {final_area_span:.3f} cm²")
            print(f"  Area std dev: {final_area_std:.3f} cm²")
            print(f"  Specification threshold: {spec_threshold} cm²")
            if final_area_span < spec_threshold:
                print("  ✓ SPECIFICATION COMPLIANT")
            else:
                print("  ⚠️  SPECIFICATION VIOLATION")
            print(f"  Ratio variance: {final_ratio_variance:.4f}")
            print(f"{'=' * 50}\n")

        return best_cuts

    def _fallback_single_solution(
        self, target_area: float, target_ratio: float
    ) -> List[Tuple[Point, Point]]:
        """Fallback to the original single-solution approach if Stage 1 fails."""
        print("Using fallback single-solution approach...")

        # Use the original greedy approach from Player 10
        cake_copy = self.cake.copy()
        all_cuts = []
        pieces_queue = [(cake_copy.exterior_shape, self.children)]
        cut_number = 0

        while cut_number < self.children - 1:
            # Find piece to cut (same logic as original)
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

            pieces_queue.pop(cutting_index)

            # Try to find a good cut (simplified version)
            min_split = 1
            max_split = max(1, cutting_num_children // 2)

            best_cut = None
            best_score = float("inf")

            # Try a few random combinations
            for _ in range(50):  # Fewer attempts for fallback
                split_children = random.randint(min_split, max_split)
                angle = random.uniform(0, 180)

                remaining_children = cutting_num_children - split_children
                target_cut_area = target_area * split_children

                position = self.binary_search(cutting_piece, target_cut_area, angle)
                if position is None:
                    continue

                cut_line = self.find_line(position, cutting_piece, angle)
                cut_points = self.find_cuts(cut_line, cutting_piece)
                if cut_points is None:
                    continue

                from_p, to_p = cut_points

                # Simulate the cut
                test_pieces = split(cutting_piece, cut_line)
                if len(test_pieces.geoms) != 2:
                    continue

                p1, p2 = test_pieces.geoms
                if abs(p1.area - target_cut_area) < abs(p2.area - target_cut_area):
                    small_piece, large_piece = p1, p2
                else:
                    small_piece, large_piece = p2, p1

                # Simple scoring
                size_error = abs(small_piece.area - target_cut_area)
                score = size_error

                if score < best_score:
                    best_score = score
                    best_cut = (
                        from_p,
                        to_p,
                        small_piece,
                        large_piece,
                        split_children,
                        remaining_children,
                    )

            if best_cut is None:
                print(f"Failed to find cut {cut_number + 1}")
                break

            (
                from_p,
                to_p,
                small_piece,
                large_piece,
                split_children,
                remaining_children,
            ) = best_cut

            # Make the cut
            try:
                cake_copy.cut(from_p, to_p)
                all_cuts.append((from_p, to_p))
                cut_number += 1

                pieces_queue.append((small_piece, split_children))
                pieces_queue.append((large_piece, remaining_children))
            except Exception:
                print(f"Cut {cut_number + 1} failed")
                break

        return all_cuts
