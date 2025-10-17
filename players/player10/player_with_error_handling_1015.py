from shapely.geometry import LineString, Point, Polygon
import math
import random
import time
from statistics import stdev, StatisticsError

from concurrent.futures import ProcessPoolExecutor, as_completed

from players.player import Player
from src.cake import Cake
from shapely.ops import split

COMPUTATION_RATIO = 8
PHRASE_ONE_TOTAL_ATTEMPS = 90 * 9 * COMPUTATION_RATIO
PHRASE_TWO_TOTAL_ATTEMPS = 360 * 9 * COMPUTATION_RATIO
PHRASE_THREE_TOTAL_ATTEMPS = 60 * 9 * COMPUTATION_RATIO
PHRASE_THREE_STEP = 2.5

# Error handling and retry constants
SIZE_SPAN_THRESHOLD = 0.5  # Maximum allowed area span (same as final evaluation)
RATIO_VARIANCE_THRESHOLD = 3  # Maximum allowed crust ratio variance
MIN_COMPUTATION_RATIO = 0.5  # Minimum computation ratio before giving up
DEFAULT_MAX_REPEAT_TIMES = 20  # Maximum number of retry attempts
TIME_LIMIT_SECONDS = 60 * 4.5  # Maximum time limit before timeout
DEFAULT_MINI_TIME = 60 * 1.5  # Time threshold for computation ratio decay


class Player10(Player):
    def __init__(
        self,
        children: int,
        cake: Cake,
        cake_path: str | None,
        phrase_three_attempts: int = 180,
        num_of_processes: int = 8,
        max_repeat_times: int = DEFAULT_MAX_REPEAT_TIMES,
        mini_time: int = DEFAULT_MINI_TIME,
    ) -> None:
        super().__init__(children, cake, cake_path)
        # Binary search tolerance: area within 0.5 cm² of target
        self.target_area_tolerance = 0.0001
        # Number of different angles to try in phase 1 (more attempts = better for complex shapes)
        self.phrase_one_attempts = PHRASE_ONE_TOTAL_ATTEMPS // (children - 1)
        # Number of different angles to try in phase 2 (more attempts = better for complex shapes)
        self.phrase_two_attempts = PHRASE_TWO_TOTAL_ATTEMPS // (children - 1)
        # Number of different angles to try in phase 3 (fine-grained search)
        self.phrase_three_attempts = PHRASE_THREE_TOTAL_ATTEMPS // (children - 1)
        # Number of processes for concurrent search
        # self.num_of_processes = min(num_of_processes, mp.cpu_count())
        self.num_of_processes = num_of_processes
        # Maximum number of retry attempts before giving up
        self.max_repeat_times = max_repeat_times
        # Time threshold for computation ratio decay
        self.mini_time = mini_time

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

        return self._get_cuts_with_retry(target_area, target_ratio)

    def _is_result_good_enough(self, areas, ratios):
        """Check if the final result meets the quality thresholds."""
        if len(areas) < 2:
            return False

        size_span = max(areas) - min(areas)
        if size_span > SIZE_SPAN_THRESHOLD:
            return False

        # Check ratio variance if we have multiple ratios
        if len(ratios) > 1:
            try:
                ratio_variance = (
                    stdev(ratios) * 100
                )  # Scale to match evaluation display
                if ratio_variance > RATIO_VARIANCE_THRESHOLD:
                    return False
            except (StatisticsError, ValueError, TypeError):
                # If stdev fails, check if all ratios are within acceptable range (scaled)
                ratio_range = (max(ratios) - min(ratios)) * 100
                if ratio_range > RATIO_VARIANCE_THRESHOLD:
                    return False

        return True

    def _get_cuts_with_retry(
        self, target_area: float, target_ratio: float
    ) -> list[tuple[Point, Point]]:
        """Main cutting logic with retry mechanism and decaying computation ratio."""
        start_time = time.time()
        current_computation_ratio = COMPUTATION_RATIO
        original_computation_ratio = COMPUTATION_RATIO
        current_num_processes = self.num_of_processes
        attempt = 0

        # Track all results for final selection
        all_results = []  # List of (cuts, areas, ratios, size_span, ratio_variance_scaled)

        while attempt < self.max_repeat_times:
            # Check for timeout before starting attempt
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_LIMIT_SECONDS:
                print(f"⏰ TIMEOUT after {elapsed_time:.1f}s, selecting best result...")
                return self._select_best_result(all_results) if all_results else []

            attempt += 1
            round_start_time = time.time()
            print(f"\n{'=' * 60}")
            print(
                f"ATTEMPT {attempt}/{self.max_repeat_times} - Computation Ratio: {current_computation_ratio}, Processes: {current_num_processes}"
            )
            print(f"{'=' * 60}")

            # Set different random seed for each attempt
            random.seed(42 + attempt)

            try:
                # Update computation attempts based on current ratio (following original logic)
                base_phrase_one = PHRASE_ONE_TOTAL_ATTEMPS // (self.children - 1)
                base_phrase_two = PHRASE_TWO_TOTAL_ATTEMPS // (self.children - 1)
                base_phrase_three = PHRASE_THREE_TOTAL_ATTEMPS // (self.children - 1)

                self.phrase_one_attempts = int(
                    base_phrase_one * current_computation_ratio / COMPUTATION_RATIO
                )
                self.phrase_two_attempts = int(
                    base_phrase_two * current_computation_ratio / COMPUTATION_RATIO
                )
                self.phrase_three_attempts = int(
                    base_phrase_three * current_computation_ratio / COMPUTATION_RATIO
                )

                # Ensure minimum attempts
                self.phrase_one_attempts = max(10, self.phrase_one_attempts)
                self.phrase_two_attempts = max(10, self.phrase_two_attempts)
                self.phrase_three_attempts = max(5, self.phrase_three_attempts)

                # Run the algorithm
                all_cuts = self._greedy_ratio_angle_cutting(
                    target_area, target_ratio, current_num_processes
                )

                # Check if we got a complete result
                if len(all_cuts) < self.children - 1:
                    print(
                        f"Incomplete result: got {len(all_cuts)} cuts, need {self.children - 1}"
                    )
                    # Still add incomplete results to the list for potential selection
                    all_results.append((all_cuts, [], [], float("inf"), float("inf")))

                    # Time-based computation ratio decay logic for incomplete results
                    round_time = time.time() - round_start_time
                    current_elapsed_time = time.time() - start_time
                    time_threshold = self.mini_time - current_elapsed_time

                    if current_elapsed_time > TIME_LIMIT_SECONDS:
                        print(
                            f"⏰ TIMEOUT after {current_elapsed_time:.1f}s during incomplete result, selecting best result..."
                        )
                        return (
                            self._select_best_result(all_results)
                            if all_results
                            else all_cuts
                        )
                    elif (
                        attempt >= self.max_repeat_times
                        and current_computation_ratio <= MIN_COMPUTATION_RATIO
                    ):
                        print(
                            "Max attempts reached and minimum computation ratio reached, selecting best result..."
                        )
                        return (
                            self._select_best_result(all_results)
                            if all_results
                            else all_cuts
                        )
                    elif round_time >= time_threshold:
                        # Current round took too long even with incomplete result, decay computation ratio for next round
                        print(
                            f"Incomplete round took {round_time:.1f}s >= threshold {time_threshold:.1f}s, decaying computation ratio"
                        )
                        old_ratio = current_computation_ratio
                        current_computation_ratio = max(
                            MIN_COMPUTATION_RATIO, current_computation_ratio / 2
                        )
                        # Decay processes only when computation ratio decays by factor of 4
                        if (
                            old_ratio >= original_computation_ratio / 2
                            and current_computation_ratio
                            < original_computation_ratio / 4
                        ):
                            current_num_processes = max(1, current_num_processes // 2)
                    # Otherwise, keep same computation ratio for next round

                    continue

                # Evaluate the final result
                try:
                    cake_copy = self.cake.copy()
                    for cut in all_cuts:
                        cake_copy.cut(cut[0], cut[1])

                    pieces = cake_copy.get_pieces()
                    areas = [p.area for p in pieces]
                    ratios = cake_copy.get_piece_ratios()

                    # Calculate metrics
                    size_span = max(areas) - min(areas) if areas else float("inf")
                    ratio_variance_scaled = 0.0
                    if len(ratios) > 1:
                        try:
                            ratio_variance_scaled = stdev(ratios) * 100
                        except (StatisticsError, ValueError, TypeError):
                            ratio_variance_scaled = (max(ratios) - min(ratios)) * 100

                    # Store this result
                    all_results.append(
                        (all_cuts, areas, ratios, size_span, ratio_variance_scaled)
                    )

                    # Print result quality info
                    if self._is_result_good_enough(areas, ratios):
                        print(
                            f"✓ Good result found on attempt {attempt} (continuing to explore...)"
                        )
                    else:
                        print(f"✗ Result not good enough on attempt {attempt}")

                    print(
                        f"  Size span: {size_span:.2f} (threshold: {SIZE_SPAN_THRESHOLD})"
                    )
                    print(
                        f"  Ratio variance: {ratio_variance_scaled:.2f} (threshold: {RATIO_VARIANCE_THRESHOLD})"
                    )

                    # Check if we found a perfect result (both metrics are 0.00)
                    size_span_rounded = round(size_span, 2)
                    ratio_variance_rounded = round(ratio_variance_scaled, 2)
                    print(
                        f"DEBUG: size_span={size_span:.6f} -> {size_span_rounded}, ratio_variance={ratio_variance_scaled:.6f} -> {ratio_variance_rounded}"
                    )

                    if size_span_rounded == 0.00 and ratio_variance_rounded == 0.00:
                        print(
                            "🎯 PERFECT result found! Size span and ratio variance both 0.00, returning immediately!"
                        )
                        return all_cuts

                    # Time-based computation ratio decay logic
                    round_time = time.time() - round_start_time
                    current_elapsed_time = time.time() - start_time
                    time_threshold = self.mini_time - current_elapsed_time
                    if current_elapsed_time > TIME_LIMIT_SECONDS:
                        print(
                            f"⏰ TIMEOUT after {current_elapsed_time:.1f}s, selecting best result..."
                        )
                        return self._select_best_result(all_results)
                    elif (
                        attempt >= self.max_repeat_times
                        and current_computation_ratio <= MIN_COMPUTATION_RATIO
                    ):
                        print(
                            "Max attempts reached and minimum computation ratio reached, selecting best result..."
                        )
                        return self._select_best_result(all_results)
                    elif round_time >= time_threshold:
                        # Current round took too long, decay computation ratio for next round
                        print(
                            f"Round took {round_time:.1f}s >= threshold {time_threshold:.1f}s, decaying computation ratio"
                        )
                        old_ratio = current_computation_ratio
                        current_computation_ratio = max(
                            MIN_COMPUTATION_RATIO, current_computation_ratio / 2
                        )
                        # Decay processes only when computation ratio decays by factor of 4
                        if (
                            old_ratio >= original_computation_ratio / 2
                            and current_computation_ratio
                            < original_computation_ratio / 4
                        ):
                            current_num_processes = max(1, current_num_processes // 2)
                    # Otherwise, keep same computation ratio for next round

                    continue

                # Evaluate the final result
                except Exception as e:
                    print(f"✗ Error evaluating result on attempt {attempt}: {e}")
                    # Add a placeholder result for failed attempts
                    all_results.append(([], [], [], float("inf"), float("inf")))

                    # Time-based computation ratio decay logic for errors
                    round_time = time.time() - round_start_time
                    error_elapsed_time = time.time() - start_time
                    time_threshold = self.mini_time - error_elapsed_time

                    if error_elapsed_time > TIME_LIMIT_SECONDS:
                        print(
                            f"⏰ TIMEOUT after {error_elapsed_time:.1f}s during error, selecting best result..."
                        )
                        return (
                            self._select_best_result(all_results) if all_results else []
                        )
                    elif (
                        attempt >= self.max_repeat_times
                        and current_computation_ratio <= MIN_COMPUTATION_RATIO
                    ):
                        print(
                            "Max attempts reached and minimum computation ratio reached, selecting best result..."
                        )
                        return (
                            self._select_best_result(all_results) if all_results else []
                        )
                    elif round_time >= time_threshold:
                        # Current round took too long even with error, decay computation ratio for next round
                        print(
                            f"Round with error took {round_time:.1f}s >= threshold {time_threshold:.1f}s, decaying computation ratio"
                        )
                        old_ratio = current_computation_ratio
                        current_computation_ratio = max(
                            MIN_COMPUTATION_RATIO, current_computation_ratio / 2
                        )
                        # Decay processes only when computation ratio decays by factor of 4
                        if (
                            old_ratio >= original_computation_ratio / 2
                            and current_computation_ratio
                            < original_computation_ratio / 4
                        ):
                            current_num_processes = max(1, current_num_processes // 2)
                    # Otherwise, keep same computation ratio for next round

                    continue

            except Exception as e:
                print(f"✗ Error during attempt {attempt}: {e}")
                # Add a placeholder result for failed attempts
                all_results.append(([], [], [], float("inf"), float("inf")))

                # Time-based computation ratio decay logic for algorithm errors
                round_time = time.time() - round_start_time
                algorithm_error_elapsed_time = time.time() - start_time
                time_threshold = self.mini_time - algorithm_error_elapsed_time

                if algorithm_error_elapsed_time > TIME_LIMIT_SECONDS:
                    print(
                        f"⏰ TIMEOUT after {algorithm_error_elapsed_time:.1f}s during algorithm error, selecting best result..."
                    )
                    return self._select_best_result(all_results) if all_results else []
                elif (
                    attempt >= self.max_repeat_times
                    and current_computation_ratio <= MIN_COMPUTATION_RATIO
                ):
                    print(
                        "Max attempts reached and minimum computation ratio reached, selecting best result..."
                    )
                    return self._select_best_result(all_results) if all_results else []
                elif round_time >= time_threshold:
                    # Current round took too long even with error, decay computation ratio for next round
                    print(
                        f"Round with error took {round_time:.1f}s >= threshold {time_threshold:.1f}s, decaying computation ratio"
                    )
                    old_ratio = current_computation_ratio
                    current_computation_ratio = max(
                        MIN_COMPUTATION_RATIO, current_computation_ratio / 2
                    )
                    # Decay processes only when computation ratio decays by factor of 4
                    if (
                        old_ratio >= original_computation_ratio / 2
                        and current_computation_ratio < original_computation_ratio / 4
                    ):
                        current_num_processes = max(1, current_num_processes // 2)
                # Otherwise, keep same computation ratio for next round

                continue

        # Final timeout check before selecting best result
        final_elapsed_time = time.time() - start_time
        if final_elapsed_time > TIME_LIMIT_SECONDS:
            print(
                f"⏰ TIMEOUT after {final_elapsed_time:.1f}s at end, selecting best result..."
            )
            return self._select_best_result(all_results) if all_results else []

        print(
            f"Max attempts ({self.max_repeat_times}) reached, selecting best result..."
        )
        return self._select_best_result(all_results) if all_results else []

    def _select_best_result(self, all_results):
        """Select the best result from all attempts based on quality criteria."""
        if not all_results:
            return []

        # Filter out incomplete/failed results (empty cuts)
        complete_results = [
            result for result in all_results if result[0]
        ]  # cuts is not empty

        if not complete_results:
            # If all results are incomplete, return the first one anyway
            return all_results[0][0] if all_results else []

        # Separate results into those with valid size span and those without
        valid_size_span_results = []
        invalid_size_span_results = []

        for result in complete_results:
            cuts, areas, ratios, size_span, ratio_variance_scaled = result
            if size_span <= SIZE_SPAN_THRESHOLD:
                valid_size_span_results.append(result)
            else:
                invalid_size_span_results.append(result)

        # First priority: results with valid size span, choose minimum ratio variance
        if valid_size_span_results:
            best_result = min(
                valid_size_span_results, key=lambda x: x[4]
            )  # Sort by ratio_variance_scaled
            cuts, areas, ratios, size_span, ratio_variance_scaled = best_result
            print(
                f"Selected best result: size_span={size_span:.2f}, ratio_variance={ratio_variance_scaled:.2f}"
            )
            return cuts

        # Second priority: if no valid size span, use minimum size span (original scoring)
        if invalid_size_span_results:
            best_result = min(
                invalid_size_span_results, key=lambda x: x[3]
            )  # Sort by size_span
            cuts, areas, ratios, size_span, ratio_variance_scaled = best_result
            print(
                f"No valid size span found, selected minimum size span: {size_span:.2f}"
            )
            return cuts

        # This shouldn't happen, but fallback to first complete result
        return complete_results[0][0]

    def _greedy_ratio_angle_cutting(
        self, target_area: float, target_ratio: float, num_processes: int = 8
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
            for _ in range(self.phrase_one_attempts // 2):
                split_children = random.randint(min_split, max_split)
                angle = random.uniform(0, 180)
                attempts_to_try.append((split_children, angle, "phase1"))

            # Process Phase 1 attempts concurrently
            print(
                f"  Phase 1: Processing {len(attempts_to_try)} attempts across {num_processes} processes..."
            )

            # Split attempts into batches for each process
            batch_size = max(1, len(attempts_to_try) // num_processes)
            batches = [
                attempts_to_try[i : i + batch_size]
                for i in range(0, len(attempts_to_try), batch_size)
            ]

            # Use ProcessPoolExecutor for concurrent processing
            with ProcessPoolExecutor(
                max_workers=max(1, num_processes // 4)
            ) as executor:
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
                    f"  Phase 2: Trying {self.phrase_two_attempts} more angles with best ratio across {num_processes} processes..."
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
                batch_size = max(1, len(phase2_attempts_to_try) // num_processes)
                batches = [
                    phase2_attempts_to_try[i : i + batch_size]
                    for i in range(0, len(phase2_attempts_to_try), batch_size)
                ]

                # Use ProcessPoolExecutor for concurrent processing
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
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

            # Phase 3: Fine-grained search around the best angle
            if best_cut is not None:
                best_angle = best_cut[6]  # Extract angle from best_cut tuple

                print(
                    f"  Phase 2 complete. Best angle: {best_angle:.1f}° with score {best_score:.3f}"
                )
                print(
                    f"  Phase 3: Fine-grained search around {best_angle:.1f}° with {self.phrase_three_attempts} attempts..."
                )

                # Calculate angle step from phase 2
                angle_step_phase2 = 360.0 / self.phrase_two_attempts

                # Define search range: best_angle +/- 2 * angle_step_phase2
                search_range = PHRASE_THREE_STEP * angle_step_phase2
                angle_min = max(0, best_angle - search_range)
                angle_max = min(360, best_angle + search_range)

                print(f"  Search range: {angle_min:.1f}° to {angle_max:.1f}°")

                # Generate fine-grained angles uniformly in the search range
                if self.phrase_three_attempts > 1:
                    fine_angle_step = (angle_max - angle_min) / (
                        self.phrase_three_attempts - 1
                    )
                    phase3_angles = [
                        angle_min + i * fine_angle_step
                        for i in range(self.phrase_three_attempts)
                    ]
                else:
                    phase3_angles = [best_angle]

                # Process Phase 3 attempts concurrently
                phase3_attempts_to_try = [
                    (best_ratio_from_phase1, angle, "phase3") for angle in phase3_angles
                ]

                # Split phase 3 attempts into batches for each process
                batch_size = max(1, len(phase3_attempts_to_try) // num_processes)
                batches = [
                    phase3_attempts_to_try[i : i + batch_size]
                    for i in range(0, len(phase3_attempts_to_try), batch_size)
                ]

                # Use ProcessPoolExecutor for concurrent processing
                with ProcessPoolExecutor(
                    max_workers=max(1, num_processes // 4)
                ) as executor:
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
                            print(f"    Error in phase 3 batch processing: {e}")
                            continue

                print(
                    f"    Phase 3 complete. Final best angle: {best_cut[6]:.1f}° with score {best_score:.3f}"
                )
            else:
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
            ratio_variance = stdev(ratios) * 100
        print(f"  Variance: {ratio_variance:.2f}")
        print(
            f"  Min: {min(ratios):.3f}, Max: {max(ratios):.3f}, Span: {max(ratios) - min(ratios):.3f}"
        )
        print(f"{'=' * 50}\n")

        return all_cuts
