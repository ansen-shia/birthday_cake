from shapely.geometry import LineString, Point, Polygon
import math
import random
from statistics import stdev
from concurrent.futures import ProcessPoolExecutor, as_completed

from players.player import Player
from src.cake import Cake
from shapely.ops import split

COMPUTATION_RATIO = 6
PHRASE_ONE_TOTAL_ATTEMPS = 60 * 9 * COMPUTATION_RATIO
PHRASE_TWO_TOTAL_ATTEMPS = 360 * 9 * COMPUTATION_RATIO


class Player10(Player):
    def __init__(
        self,
        children: int,
        cake: Cake,
        cake_path: str | None,
        # phrase_one_attempts: int = 90 * 8,
        # phrase_two_attempts: int = 360 * 8,
        num_of_processes: int = 8,
    ) -> None:
        super().__init__(children, cake, cake_path)
        # Binary search tolerance: area within 0.5 cm² of target
        self.target_area_tolerance = 0.0001
        # Number of different angles to try in phase 1 (more attempts = better for complex shapes)
        self.phrase_one_attempts = PHRASE_ONE_TOTAL_ATTEMPS // (children - 1)
        # Number of different angles to try in phase 2 (more attempts = better for complex shapes)
        self.phrase_two_attempts = PHRASE_TWO_TOTAL_ATTEMPS // (children - 1)
        # # Number of processes for concurrent search
        # self.num_of_processes = min(num_of_processes, mp.cpu_count())
        self.num_of_processes = num_of_processes

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

    def _evaluate_attempt(self, args):
        """Helper method for multiprocessing - evaluate a single attempt"""
        (
            split_children,
            angle,
            cutting_piece,
            cutting_num_children,
            target_area,
            target_ratio,
            phase,
        ) = args

        remaining_children = cutting_num_children - split_children
        target_cut_area = target_area * split_children

        # Find the cut position using binary search
        position = self.binary_search(cutting_piece, target_cut_area, angle)
        if position is None:
            return None

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

        # Get crust ratios
        ratio1 = self.cake.get_piece_ratio(small_piece)
        ratio2 = self.cake.get_piece_ratio(large_piece)

        # Score this cut
        size_error = abs(small_piece.area - target_cut_area)
        ratio_error = abs(ratio1 - target_ratio) + abs(ratio2 - target_ratio)
        score = size_error * 3.0 + ratio_error * 1.0

        return {
            "score": score,
            "cut_points": (from_p, to_p),
            "small_piece": small_piece,
            "large_piece": large_piece,
            "ratio1": ratio1,
            "ratio2": ratio2,
            "angle": angle,
            "split_children": split_children,
            "remaining_children": remaining_children,
            "size_error": size_error,
            "ratio_error": ratio_error,
        }

    def _process_batch(
        self, batch_args, cutting_piece, cutting_num_children, target_area, target_ratio
    ):
        """Process a batch of attempts and return the best result"""
        best_result = None
        best_score = float("inf")

        for args in batch_args:
            result = self._evaluate_attempt(
                (
                    args[0],
                    args[1],
                    cutting_piece,
                    cutting_num_children,
                    target_area,
                    target_ratio,
                    args[2],
                )
            )
            if result and result["score"] < best_score:
                best_score = result["score"]
                best_result = result

        return best_result

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic - greedy approach with random (ratio, angle) pairs"""
        print(f"__________Cutting for {self.children} children_______")

        target_area = self.cake.get_area() / self.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area
        print(f"TARGET AREA: {target_area:.2f} cm²")
        print(f"TARGET CRUST RATIO: {target_ratio:.3f}")
        print("Strategy: Greedy cutting with random ratio+angle exploration\n")

        return self._greedy_ratio_angle_cutting(target_area, target_ratio)

    def _greedy_ratio_angle_cutting(
        self, target_area: float, target_ratio: float
    ) -> list[tuple[Point, Point]]:
        """
        TRUE divide-and-conquer approach:
        - Track how many children each piece is for (e.g., 1/n, 2/n, 3/n...)
        - Iteratively divide pieces until all are for 1 child
        - For each piece with n>1 children, try random (split_ratio, angle) pairs
        """
        cake_copy = self.cake.copy()
        all_cuts = []

        # Initialize: the whole cake is for all children
        # pieces_queue: list of (piece_polygon, num_children_for_this_piece)
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

            print(f"\n=== Cut {cut_number + 1}/{self.children - 1} ===")
            print(
                f"Dividing piece for {cutting_num_children} children (area: {cutting_piece.area:.2f})"
            )

            # Try different split ratios: split n children into (k, n-k)
            # where k ranges from 1 to floor(n/2) for balanced divide-and-conquer
            min_split = 1
            max_split = max(1, cutting_num_children // 2)
            print(
                f"Exploring split ratios: 1/{cutting_num_children} to {max_split}/{cutting_num_children}"
            )

            # Two-phase strategy:
            # Phase 1: Try all split ratios with cardinal angles + random sampling to find best ratio
            # Phase 2: Focus on best split ratio, only vary angles
            cardinal_angles = [0, 90, 180, 270]

            best_cut = None
            best_score = float("inf")
            best_split_ratio = None
            valid_attempts = 0

            # Track best score for each split ratio
            split_ratio_scores = {}
            for split_children in range(min_split, max_split + 1):
                split_ratio_scores[split_children] = float("inf")

            # Build list of (split_ratio, angle) to try for Phase 1
            attempts_to_try = []

            # First: Try all split ratios with all cardinal angles
            for split_children in range(min_split, max_split + 1):
                for angle in cardinal_angles:
                    attempts_to_try.append((split_children, angle, "phase1"))

            # Phase 1: Random sample all split ratios (first half)
            for _ in range(self.phrase_one_attempts):
                split_children = random.randint(min_split, max_split)
                angle = random.uniform(0, 180)
                attempts_to_try.append((split_children, angle, "phase1"))

            # Process Phase 1 attempts concurrently
            print(
                f"  Phase 1: Processing {len(attempts_to_try)} attempts across {self.num_of_processes} processes..."
            )

            # Split attempts into batches for each process
            batch_size = max(1, len(attempts_to_try) // self.num_of_processes)
            batches = [
                attempts_to_try[i : i + batch_size]
                for i in range(0, len(attempts_to_try), batch_size)
            ]

            # Use ProcessPoolExecutor for concurrent processing
            with ProcessPoolExecutor(max_workers=self.num_of_processes) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(
                        self._process_batch,
                        batch,
                        cutting_piece,
                        cutting_num_children,
                        target_area,
                        target_ratio,
                    ): batch
                    for batch in batches
                }

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    try:
                        result = future.result()
                        if result:
                            valid_attempts += 1

                            # Update best scores
                            split_children = result["split_children"]
                            if result["score"] < split_ratio_scores[split_children]:
                                split_ratio_scores[split_children] = result["score"]

                            if result["score"] < best_score:
                                best_score = result["score"]
                                best_cut = (
                                    result["cut_points"][0],
                                    result["cut_points"][1],
                                    result["small_piece"],
                                    result["large_piece"],
                                    result["ratio1"],
                                    result["ratio2"],
                                    result["angle"],
                                )
                                best_split_ratio = (
                                    split_children,
                                    result["remaining_children"],
                                )

                    except Exception as e:
                        print(f"    Error in batch processing: {e}")
                        continue

            # Phase 2: Use the best split ratio found, only vary angles
            if split_ratio_scores:
                # Find the split ratio with the best score
                best_ratio_from_phase1 = min(
                    split_ratio_scores.keys(), key=lambda k: split_ratio_scores[k]
                )

                print(
                    f"  Phase 1 complete. Best split ratio: {best_ratio_from_phase1}/{cutting_num_children}"
                )
                print(
                    f"  Phase 2: Trying {self.phrase_two_attempts} more angles with best ratio across {self.num_of_processes} processes..."
                )

                # Generate angles for phase 2
                angle_step = 360.0 / self.phrase_two_attempts
                phase2_angles = [
                    i * angle_step for i in range(self.phrase_two_attempts)
                ]

                # Process Phase 2 attempts concurrently
                phase2_attempts_to_try = [
                    (best_ratio_from_phase1, angle, "phase2") for angle in phase2_angles
                ]

                # Split phase 2 attempts into batches for each process
                batch_size = max(
                    1, len(phase2_attempts_to_try) // self.num_of_processes
                )
                batches = [
                    phase2_attempts_to_try[i : i + batch_size]
                    for i in range(0, len(phase2_attempts_to_try), batch_size)
                ]

                # Use ProcessPoolExecutor for concurrent processing
                with ProcessPoolExecutor(max_workers=self.num_of_processes) as executor:
                    # Submit all batches
                    future_to_batch = {
                        executor.submit(
                            self._process_batch,
                            batch,
                            cutting_piece,
                            cutting_num_children,
                            target_area,
                            target_ratio,
                        ): batch
                        for batch in batches
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_batch):
                        try:
                            result = future.result()
                            if result:
                                valid_attempts += 1

                                if result["score"] < best_score:
                                    best_score = result["score"]
                                    best_cut = (
                                        result["cut_points"][0],
                                        result["cut_points"][1],
                                        result["small_piece"],
                                        result["large_piece"],
                                        result["ratio1"],
                                        result["ratio2"],
                                        result["angle"],
                                    )
                                    best_split_ratio = (
                                        result["split_children"],
                                        result["remaining_children"],
                                    )

                        except Exception as e:
                            print(f"    Error in phase 2 batch processing: {e}")
                            continue

            print(f"    Best cut found with score {best_score:.3f}")
            if best_cut is None:
                print(
                    f"  No valid cut found after {len(attempts_to_try) + len(phase2_attempts_to_try)} attempts!"
                )
                # Put the piece back for now (shouldn't happen often)
                pieces_queue.append((cutting_piece, cutting_num_children))
                continue

            from_p, to_p, small_piece, large_piece, ratio1, ratio2, used_angle = (
                best_cut
            )
            split_children, remaining_children = best_split_ratio

            # Make the cut on the actual cake
            cake_copy.cut(from_p, to_p)
            all_cuts.append((from_p, to_p))
            cut_number += 1

            # Add the two new pieces to the queue with their child counts
            pieces_queue.append((small_piece, split_children))
            pieces_queue.append((large_piece, remaining_children))

            # Print info
            print(f"  Best cut (tried {valid_attempts} valid attempts)")
            print(
                f"  Split ratio: {split_children}/{cutting_num_children} and {remaining_children}/{cutting_num_children}, angle={used_angle:.1f}°"
            )
            print(
                f"  Piece 1 ({split_children} children): size={small_piece.area:.2f} (target={split_children * target_area:.2f}), crust_ratio={ratio1:.3f}"
            )
            print(
                f"  Piece 2 ({remaining_children} children): size={large_piece.area:.2f} (target={remaining_children * target_area:.2f}), crust_ratio={ratio2:.3f}"
            )

            # Show current queue status
            total_in_queue = sum(nc for _, nc in pieces_queue)
            print(
                f"  Queue: {len(pieces_queue)} pieces for {total_in_queue} total children"
            )

        # Final summary
        print(f"\n{'=' * 50}")
        print(f"FINAL RESULT: {len(all_cuts)}/{self.children - 1} cuts completed")

        pieces = cake_copy.get_pieces()
        areas = [p.area for p in pieces]
        ratios = cake_copy.get_piece_ratios()

        print(f"\nPiece areas: {[f'{a:.2f}' for a in sorted(areas)]}")
        print(
            f"  Min: {min(areas):.2f}, Max: {max(areas):.2f}, Span: {max(areas) - min(areas):.2f}"
        )

        print(f"\nCrust ratios: {[f'{r:.3f}' for r in ratios]}")
        if len(ratios) > 1:
            ratio_variance = stdev(ratios)
        print(f"  Variance: {ratio_variance:.4f}")
        print(
            f"  Min: {min(ratios):.3f}, Max: {max(ratios):.3f}, Span: {max(ratios) - min(ratios):.3f}"
        )
        print(f"{'=' * 50}\n")

        return all_cuts
