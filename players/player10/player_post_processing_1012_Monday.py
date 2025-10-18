from shapely.geometry import LineString, Point, Polygon
import math
import random
from statistics import stdev
import copy
import yaml
from pathlib import Path

from players.player import Player
from src.cake import Cake
from shapely.ops import split

# Default parameters (can be overridden by YAML file)
NUMBER_ATTEMPS = 30
GRADIENT_DESCENT_ITERATIONS = 30
LEARNING_RATE_ANGLE = 0.2
LEARNING_RATE_POSITION = 0.01
FINITE_DIFF_EPSILON = 0.01


class Player10(Player):
    def __init__(
        self,
        children: int,
        cake: Cake,
        cake_path: str | None,
        num_angle_attempts: int = None,
        config_path: str = None,
    ) -> None:
        super().__init__(children, cake, cake_path)

        # Load configuration from YAML file
        self.config = self._load_config(config_path)

        # Binary search tolerance: area within 0.5 cm² of target
        self.target_area_tolerance = 0.005

        # Number of different angles to try (more attempts = better for complex shapes)
        if num_angle_attempts is not None:
            self.num_angle_attempts = num_angle_attempts
        else:
            self.num_angle_attempts = self.config["number_attempts"]

        # Gradient descent parameters
        self.gd_iterations = self.config["gradient_descent"]["n_iterations"]
        self.gd_position_lr = self.config["gradient_descent"]["position_learning_rate"]
        self.gd_angle_lr = self.config["gradient_descent"]["angle_learning_rate"]
        self.gd_epsilon = self.config["gradient_descent"]["finite_diff_epsilon"]

        # Test mode
        self.test_mode = self.config["test_mode"]

        # Visualization settings
        if self.test_mode:
            self.viz_config = self.config["visualization"]
            self.output_dir = Path(__file__).parent / self.viz_config["output_dir"]
            self.output_dir.mkdir(exist_ok=True)

            # Import matplotlib only in test mode
            try:
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend
                import matplotlib.pyplot as plt

                self.plt = plt
            except ImportError:
                print("Warning: matplotlib not available, disabling visualizations")
                self.test_mode = False

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Use default config
            config_path = Path(__file__).parent / "parameters" / "default.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            print(
                f"Warning: Config file {config_path} not found, using hardcoded defaults"
            )
            return {
                "number_attempts": NUMBER_ATTEMPS,
                "gradient_descent": {
                    "n_iterations": GRADIENT_DESCENT_ITERATIONS,
                    "position_learning_rate": LEARNING_RATE_POSITION,
                    "angle_learning_rate": LEARNING_RATE_ANGLE,
                    "finite_diff_epsilon": FINITE_DIFF_EPSILON,
                },
                "test_mode": False,
                "visualization": {
                    "save_plots": True,
                    "show_plots": False,
                    "output_dir": "outputs",
                    "plot_format": "png",
                    "dpi": 150,
                },
            }

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print(f"Loaded configuration from: {config_path}")
        return config

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
        """This function finds the actual points where the cut line goes through cake"""
        intersection = line.intersection(piece.boundary)

        # What is the intersections geometry? - want it to be at least two points
        if intersection.is_empty or intersection.geom_type == "Point":
            return None
        points = []
        if intersection.geom_type == "MultiPoint":
            points = list(intersection.geoms)
        elif intersection.geom_type == "LineString":
            coords = list(intersection.coords)
            points = [Point(c) for c in coords]

        # Need at least 2 points for a valid cut
        if len(points) < 2:
            return None

        # return the points where the sweeping line intersects with the cake
        return (points[0], points[1])

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

    def try_cutting_at_angle(
        self, angle: float, verbose: bool = False
    ) -> tuple[list[tuple[Point, Point]], float] | None:
        """Try cutting the cake at a specific angle and return cuts with their score.

        Args:
            angle: Angle in degrees (0-360) for cutting direction
            verbose: Whether to print debug information

        Returns:
            Tuple of (cuts, variance_score) if successful, None if failed
            Lower variance_score is better
        """
        if verbose:
            print(f"\n  Trying angle {angle:.1f} degrees...")

        target_area = self.cake.get_area() / self.children
        cuts = []
        cake_copy = self.cake.copy()

        # Try to make all n-1 cuts
        for cut_idx in range(self.children - 1):
            current_pieces = cake_copy.get_pieces()
            # Always cut the biggest piece
            cutting_piece = max(current_pieces, key=lambda pc: pc.area)

            if verbose:
                print(
                    f"    Cut {cut_idx + 1}: Cutting piece with area {cutting_piece.area:.2f}, target piece {target_area:.2f}"
                )

            # Find the best position using binary search
            position = self.binary_search(cutting_piece, target_area, angle)

            # If we can't find a position, this angle doesn't work
            if position is None:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Failed to find position")
                return None

            # Find the actual cut points
            cut_line = self.find_line(position, cutting_piece, angle)
            cut_points = self.find_cuts(cut_line, cutting_piece)

            if cut_points is None:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Failed to find cut points")
                return None

            from_p, to_p = cut_points

            # Check if the cut is valid
            is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
            if not is_valid:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Invalid - {why}")
                return None

            # Try to make the cut
            try:
                cake_copy.cut(from_p, to_p)
                cuts.append((from_p, to_p))

                # Check the resulting piece sizes
                if verbose:
                    new_pieces = cake_copy.get_pieces()
                    areas = [p.area for p in new_pieces]
                    print(
                        f"      -> Resulted in areas: {[f'{a:.2f}' for a in sorted(areas)]}"
                    )

            except Exception as e:
                if verbose:
                    print(f"    Cut {cut_idx + 1}: Exception - {e}")
                return None

        # Check if we got the right number of pieces
        if len(cake_copy.get_pieces()) != self.children:
            if verbose:
                print(
                    f"    Failed: Got {len(cake_copy.get_pieces())} pieces, expected {self.children}"
                )
            return None

        # Check piece size consistency
        areas = [p.area for p in cake_copy.get_pieces()]
        size_span = max(areas) - min(areas)

        # Per project spec: pieces within 0.5 cm² are considered same size
        # But for the sweeping algorithm, we need some tolerance
        # Use a reasonable threshold based on number of children
        max_acceptable_span = max(
            5.0, target_area * 0.15
        )  # At least 5 cm² or 15% of target

        if size_span > max_acceptable_span:
            if verbose:
                print(
                    f"    Failed: Size span {size_span:.2f} is too large (>{max_acceptable_span:.2f})"
                )
            return None

        # Calculate crust ratio variance (our score to minimize)
        ratios = cake_copy.get_piece_ratios()

        # Check if all pieces are valid (have reasonable ratios)
        if any(r < 0 or r > 1 for r in ratios):
            if verbose:
                print(f"    Failed: Invalid ratios {ratios}")
            return None

        # Calculate variance in ratios (lower is better)
        if len(ratios) > 1:
            variance = stdev(ratios)
        else:
            variance = 0.0

        if verbose:
            print(f"    Success! Variance: {variance:.4f}, Size span: {size_span:.2f}")
            print(f"    Areas: {[f'{a:.2f}' for a in sorted(areas)]}")
            print(f"    Ratios: {[f'{r:.3f}' for r in ratios]}")

        return (cuts, variance)

    def find_best_cut_at_angle(
        self,
        cutting_piece: Polygon,
        target_area: float,
        angle: float,
        cake_copy: Cake,
        is_last_cut: bool = False,
    ) -> tuple[Point, Point, float, float] | None:
        """Try cutting at a specific angle and return the cut points and resulting piece info.

        Args:
            cutting_piece: The piece to cut
            target_area: Target area for the cut piece
            angle: Angle to try
            cake_copy: Current state of the cake
            is_last_cut: Whether this is the last cut (more lenient validation)

        Returns:
            Tuple of (from_point, to_point, crust_ratio_of_new_piece, area_of_new_piece) or None if invalid
        """
        # Find the best position using binary search
        position = self.binary_search(cutting_piece, target_area, angle)

        if position is None:
            return None

        # Find the actual cut points
        cut_line = self.find_line(position, cutting_piece, angle)
        cut_points = self.find_cuts(cut_line, cutting_piece)

        if cut_points is None:
            return None

        from_p, to_p = cut_points

        # Check if the cut is valid
        is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
        if not is_valid:
            return None

        # Simulate the cut to get the new piece and its crust ratio
        test_cake = cake_copy.copy()
        try:
            # Find which piece gets created
            [p.area for p in test_cake.get_pieces()]
            test_cake.cut(from_p, to_p)
            pieces_after = test_cake.get_pieces()

            # Find the new piece (smallest one, as we're cutting off target_area)
            new_piece = min(pieces_after, key=lambda p: p.area)
            new_piece_ratio = test_cake.get_piece_ratio(new_piece)
            new_piece_area = new_piece.area

            # Validate the new piece size is reasonable
            # More lenient for last cut, or when cutting small pieces
            if is_last_cut:
                tolerance = 0.5  # Very lenient for last cut
            elif cutting_piece.area < target_area * 2:
                tolerance = 0.4  # Lenient for small pieces
            else:
                tolerance = 0.3  # Standard tolerance

            if abs(new_piece_area - target_area) > target_area * tolerance:
                return None

            return (from_p, to_p, new_piece_ratio, new_piece_area)

        except Exception:
            return None

    def compute_score(self, cake: Cake) -> float:
        """Compute scoring function: combines size span and ratio variance.
        Lower score is better.
        """
        pieces = cake.get_pieces()

        if len(pieces) != self.children:
            return float("inf")  # Invalid configuration

        # Size span (difference between largest and smallest piece)
        areas = [p.area for p in pieces]
        size_span = max(areas) - min(areas)
        target_area = self.cake.get_area() / self.children
        normalized_size_span = size_span / target_area

        # Ratio variance (standard deviation of crust ratios)
        ratios = cake.get_piece_ratios()
        if len(ratios) > 1:
            ratio_variance = stdev(ratios)
        else:
            ratio_variance = 0.0

        # Combined score (weight size more than ratio)
        score = normalized_size_span * 5.0 + ratio_variance * 2.0
        return score

    def extract_cut_parameters(self, cuts: list[tuple[Point, Point]]) -> list[dict]:
        """Extract angle and position parameters from cuts.

        Returns list of dicts with keys: 'from_p', 'to_p', 'angle', 'position', 'piece_bounds'
        """
        parameters = []
        cake_copy = self.cake.copy()

        for cut_idx, (from_p, to_p) in enumerate(cuts):
            # Get the piece being cut: find which piece contains the cut line
            current_pieces = cake_copy.get_pieces()

            # The cutting piece is the one that this line intersects
            cutting_piece = None
            cut_line = LineString([from_p, to_p])

            for piece in current_pieces:
                if piece.intersects(cut_line):
                    # Check if this is actually cutting the piece (not just touching)
                    intersection = cut_line.intersection(piece.boundary)
                    if not intersection.is_empty:
                        cutting_piece = piece
                        break

            if cutting_piece is None:
                # Fallback: use largest piece
                cutting_piece = max(current_pieces, key=lambda pc: pc.area)

            # Calculate angle from the cut line
            dx = to_p.x - from_p.x
            dy = to_p.y - from_p.y
            angle = math.degrees(math.atan2(dy, dx)) % 180  # Normalize to 0-180

            # Estimate position by projecting center of line onto sweep direction
            leftmost, lowest, rightmost, highest = cutting_piece.bounds
            center_x = (leftmost + rightmost) / 2
            center_y = (lowest + highest) / 2

            # Line center
            line_center_x = (from_p.x + to_p.x) / 2
            line_center_y = (from_p.y + to_p.y) / 2

            # Project onto sweep direction (perpendicular to angle)
            sweep_angle = math.radians(angle + 90)
            sweep_dx = line_center_x - center_x
            sweep_dy = line_center_y - center_y

            # Position as projection
            width = rightmost - leftmost
            height = highest - lowest
            max_dim = max(width, height) * 2

            if max_dim > 0:
                position = (
                    0.5
                    + (
                        sweep_dx * math.cos(sweep_angle)
                        + sweep_dy * math.sin(sweep_angle)
                    )
                    / max_dim
                )
            else:
                position = 0.5

            position = max(0.0, min(1.0, position))  # Clamp to [0, 1]

            parameters.append(
                {
                    "from_p": from_p,
                    "to_p": to_p,
                    "angle": angle,
                    "position": position,
                    "piece_bounds": cutting_piece.bounds,
                    "cutting_piece": cutting_piece,
                }
            )

            # Apply the cut for the next iteration
            try:
                cake_copy.cut(from_p, to_p)
            except Exception as e:
                print(
                    f"Warning: Failed to apply cut {cut_idx + 1} during parameter extraction: {e}"
                )
                break

        return parameters

    def apply_cuts_from_parameters(
        self, parameters: list[dict]
    ) -> tuple[list[tuple[Point, Point]], float] | None:
        """Reconstruct and apply cuts from angle/position parameters.

        Returns (cuts, score) or None if any cut is invalid.
        """
        cuts = []
        cake_copy = self.cake.copy()

        for param_idx, param in enumerate(parameters):
            # Get current pieces
            current_pieces = cake_copy.get_pieces()

            # Find the piece that best matches the original cutting piece
            # Use area and centroid to match
            original_piece = param["cutting_piece"]
            original_centroid = original_piece.centroid
            original_area = original_piece.area

            # Find best matching piece
            best_match = None
            best_match_score = float("inf")

            for piece in current_pieces:
                # Score based on area similarity and centroid distance
                area_diff = abs(piece.area - original_area)
                centroid_dist = piece.centroid.distance(original_centroid)
                match_score = area_diff + centroid_dist * 10

                if match_score < best_match_score:
                    best_match_score = match_score
                    best_match = piece

            if best_match is None:
                return None

            cutting_piece = best_match

            # Reconstruct cut from angle and position
            angle = param["angle"]
            position = param["position"]

            # Create the cut line
            cut_line = self.find_line(position, cutting_piece, angle)
            cut_points = self.find_cuts(cut_line, cutting_piece)

            if cut_points is None:
                return None

            from_p, to_p = cut_points

            # Validate the cut
            is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
            if not is_valid:
                return None

            # Apply the cut
            try:
                cake_copy.cut(from_p, to_p)
                cuts.append((from_p, to_p))
            except Exception:
                return None

        # Compute final score
        score = self.compute_score(cake_copy)
        return (cuts, score)

    def apply_cuts_directly(self, cuts: list[tuple[Point, Point]]) -> float | None:
        """Apply cuts and return score, or None if invalid."""
        cake_copy = self.cake.copy()
        for from_p, to_p in cuts:
            is_valid, why = cake_copy.cut_is_valid(from_p, to_p)
            if not is_valid:
                return None
            try:
                cake_copy.cut(from_p, to_p)
            except Exception:
                return None

        return self.compute_score(cake_copy)

    def gradient_descent_post_processing(
        self, initial_cuts: list[tuple[Point, Point]]
    ) -> list[tuple[Point, Point]]:
        """Post-process cuts using gradient descent to optimize cut point coordinates.

        Args:
            initial_cuts: The cuts from the initial algorithm

        Returns:
            Optimized cuts
        """
        print(f"\n{'=' * 70}")
        print("GRADIENT DESCENT POST-PROCESSING")
        print(f"{'=' * 70}")
        print(f"Iterations: {self.gd_iterations}")
        print(f"Learning rate: {self.gd_position_lr} per iteration")
        print(f"Finite difference epsilon: {self.gd_epsilon}")
        print(f"Test mode: {self.test_mode}\n")

        # Get initial score
        initial_score = self.apply_cuts_directly(initial_cuts)
        if initial_score is None:
            print("ERROR: Initial cuts are invalid!")
            return initial_cuts

        # Get initial metrics
        initial_cake = self.cake.copy()
        for from_p, to_p in initial_cuts:
            initial_cake.cut(from_p, to_p)
        initial_areas = [p.area for p in initial_cake.get_pieces()]
        initial_ratios = initial_cake.get_piece_ratios()
        initial_size_span = max(initial_areas) - min(initial_areas)
        initial_ratio_variance = (
            stdev(initial_ratios) if len(initial_ratios) > 1 else 0.0
        )

        print(f"Initial score: {initial_score:.6f}")
        print(f"  Size span: {initial_size_span:.4f}  cm²")
        print(f"  Ratio variance: {initial_ratio_variance:.6f}\n")

        # Convert cuts to coordinate list for optimization
        # cuts_coords[i] = [from_x, from_y, to_x, to_y] for cut i
        cuts_coords = []
        for from_p, to_p in initial_cuts:
            cuts_coords.append([from_p.x, from_p.y, to_p.x, to_p.y])

        best_cuts_coords = copy.deepcopy(cuts_coords)
        best_score = initial_score

        # Tracking for visualization
        if self.test_mode:
            metrics_history = {
                "iteration": [],
                "score": [],
                "size_span": [],
                "ratio_variance": [],
                "cuts_coords": [],  # Store cuts at each iteration for visualization
            }
            # Store initial metrics
            metrics_history["iteration"].append(0)
            metrics_history["score"].append(initial_score)
            metrics_history["size_span"].append(initial_size_span)
            metrics_history["ratio_variance"].append(initial_ratio_variance)
            metrics_history["cuts_coords"].append(copy.deepcopy(cuts_coords))

        # Gradient descent loop
        for iteration in range(self.gd_iterations):
            # Compute gradients for all cut coordinates
            gradients = []

            for cut_idx in range(len(cuts_coords)):
                cut_grad = []

                # Gradient for each coordinate (from_x, from_y, to_x, to_y)
                for coord_idx in range(4):
                    # Perturb coordinate slightly
                    coords_plus = copy.deepcopy(cuts_coords)
                    coords_minus = copy.deepcopy(cuts_coords)

                    coords_plus[cut_idx][coord_idx] += self.gd_epsilon
                    coords_minus[cut_idx][coord_idx] -= self.gd_epsilon

                    # Convert to cuts and evaluate
                    cuts_plus = [
                        (Point(c[0], c[1]), Point(c[2], c[3])) for c in coords_plus
                    ]
                    cuts_minus = [
                        (Point(c[0], c[1]), Point(c[2], c[3])) for c in coords_minus
                    ]

                    score_plus = self.apply_cuts_directly(cuts_plus)
                    score_minus = self.apply_cuts_directly(cuts_minus)

                    if score_plus is not None and score_minus is not None:
                        grad = (score_plus - score_minus) / (2 * self.gd_epsilon)
                    else:
                        grad = 0.0

                    cut_grad.append(grad)

                gradients.append(cut_grad)

            # Try different step sizes (adaptive learning rate)
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05]
            success = False

            for step_size in step_sizes:
                # Update all coordinates together
                new_cuts_coords = copy.deepcopy(cuts_coords)
                for cut_idx in range(len(cuts_coords)):
                    for coord_idx in range(4):
                        new_cuts_coords[cut_idx][coord_idx] -= (
                            step_size
                            * self.gd_position_lr
                            * gradients[cut_idx][coord_idx]
                        )

                # Convert to cuts and evaluate
                new_cuts = [
                    (Point(c[0], c[1]), Point(c[2], c[3])) for c in new_cuts_coords
                ]
                new_score = self.apply_cuts_directly(new_cuts)

                if new_score is not None and new_score < best_score:
                    improvement = best_score - new_score
                    cuts_coords = new_cuts_coords
                    best_score = new_score
                    best_cuts_coords = copy.deepcopy(cuts_coords)
                    success = True

                    # Track metrics for visualization
                    if self.test_mode:
                        # Compute current metrics
                        current_cake = self.cake.copy()
                        for from_p, to_p in new_cuts:
                            current_cake.cut(from_p, to_p)
                        current_areas = [p.area for p in current_cake.get_pieces()]
                        current_ratios = current_cake.get_piece_ratios()
                        current_size_span = max(current_areas) - min(current_areas)
                        current_ratio_variance = (
                            stdev(current_ratios) if len(current_ratios) > 1 else 0.0
                        )

                        metrics_history["iteration"].append(iteration + 1)
                        metrics_history["score"].append(new_score)
                        metrics_history["size_span"].append(current_size_span)
                        metrics_history["ratio_variance"].append(current_ratio_variance)
                        metrics_history["cuts_coords"].append(
                            copy.deepcopy(new_cuts_coords)
                        )

                    if (iteration + 1) % 10 == 0 or iteration < 5:
                        step_info = (
                            f" (step={step_size:.2f})" if step_size < 1.0 else ""
                        )
                        print(
                            f"Iteration {iteration + 1:3d}: score = {new_score:.6f} (↓ {improvement:.6f}){step_info}"
                        )
                    break

            if not success and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:3d}: No improvement found")

        # Convert best coordinates back to cuts
        final_cuts = [(Point(c[0], c[1]), Point(c[2], c[3])) for c in best_cuts_coords]

        # Print final comparison
        print(f"\n{'=' * 70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Initial score: {initial_score:.6f}")
        print(f"Final score:   {best_score:.6f}")
        print(
            f"Improvement:   {initial_score - best_score:.6f} ({(initial_score - best_score) / initial_score * 100:.2f}%)"
        )

        # Detailed comparison
        print("\n--- Initial Cuts ---")
        self._print_cut_details(initial_cuts)

        print("\n--- Final Cuts (After Optimization) ---")
        self._print_cut_details(final_cuts)

        # Coordinate changes
        print("\n--- Coordinate Changes ---")
        for i, (initial_cut, final_cut) in enumerate(zip(initial_cuts, final_cuts)):
            from_dist = final_cut[0].distance(initial_cut[0])
            to_dist = final_cut[1].distance(initial_cut[1])
            print(
                f"Cut {i + 1}: from_point moved {from_dist:.4f}, to_point moved {to_dist:.4f}"
            )

        print(f"{'=' * 70}\n")

        # Generate visualizations if in test mode
        if self.test_mode:
            if len(metrics_history["iteration"]) > 1:
                self._generate_visualizations(metrics_history, initial_cuts, final_cuts)
            else:
                print(
                    "Note: No improvements made, but generating static visualizations..."
                )
                # Add final state to history for visualization
                metrics_history["iteration"].append(self.gd_iterations)
                metrics_history["score"].append(initial_score)
                metrics_history["size_span"].append(initial_size_span)
                metrics_history["ratio_variance"].append(initial_ratio_variance)
                metrics_history["cuts_coords"].append(copy.deepcopy(cuts_coords))
                self._generate_visualizations(metrics_history, initial_cuts, final_cuts)

        return final_cuts

    def _generate_visualizations(
        self,
        metrics_history: dict,
        initial_cuts: list[tuple[Point, Point]],
        final_cuts: list[tuple[Point, Point]],
    ):
        """Generate and save visualization plots."""
        print("\nGenerating visualizations...")

        plt = self.plt

        # Create figure with multiple subplots
        plt.figure(figsize=(16, 12))

        # 1. Metrics over iterations (top row)
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(
            metrics_history["iteration"],
            metrics_history["score"],
            "o-",
            linewidth=2,
            markersize=4,
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Total Score")
        ax1.set_title("Score vs Iteration")
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(
            metrics_history["iteration"],
            metrics_history["size_span"],
            "o-",
            color="orange",
            linewidth=2,
            markersize=4,
        )
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Size Span (cm²)")
        ax2.set_title("Piece Size Span vs Iteration")
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(
            metrics_history["iteration"],
            metrics_history["ratio_variance"],
            "o-",
            color="green",
            linewidth=2,
            markersize=4,
        )
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Ratio Std Dev")
        ax3.set_title("Crust Ratio Variance vs Iteration")
        ax3.grid(True, alpha=0.3)

        # 2. Cake visualization - Initial cuts (middle row)
        ax4 = plt.subplot(3, 3, 4)
        self._plot_cake_with_cuts(ax4, initial_cuts, "Initial Cuts")

        # 3. Cake visualization - Final cuts (middle row)
        ax5 = plt.subplot(3, 3, 5)
        self._plot_cake_with_cuts(ax5, final_cuts, "Final Cuts (Optimized)")

        # 4. Cut movement visualization (middle row)
        ax6 = plt.subplot(3, 3, 6)
        self._plot_cut_movements(ax6, initial_cuts, final_cuts)

        # 5. Piece size distribution - Initial (bottom row)
        ax7 = plt.subplot(3, 3, 7)
        self._plot_piece_distribution(ax7, initial_cuts, "Initial Piece Sizes")

        # 6. Piece size distribution - Final (bottom row)
        ax8 = plt.subplot(3, 3, 8)
        self._plot_piece_distribution(ax8, final_cuts, "Final Piece Sizes")

        # 7. Crust ratio distribution (bottom row)
        ax9 = plt.subplot(3, 3, 9)
        self._plot_ratio_comparison(ax9, initial_cuts, final_cuts)

        plt.tight_layout()

        # Save the plot
        if self.viz_config["save_plots"]:
            plot_format = self.viz_config["plot_format"]
            dpi = self.viz_config["dpi"]
            output_path = (
                self.output_dir / f"gradient_descent_visualization.{plot_format}"
            )
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved visualization to: {output_path}")

        # Show the plot if configured
        if self.viz_config["show_plots"]:
            plt.show()
        else:
            plt.close()

    def _plot_cake_with_cuts(self, ax, cuts: list[tuple[Point, Point]], title: str):
        """Plot the cake outline and cuts."""
        # Plot exterior shape
        ext_x, ext_y = self.cake.exterior_shape.exterior.xy
        ax.plot(ext_x, ext_y, "k-", linewidth=2, label="Exterior")

        # Plot interior shape
        int_x, int_y = self.cake.interior_shape.exterior.xy
        ax.plot(int_x, int_y, "b-", linewidth=2, label="Interior")

        # Plot cuts
        for i, (from_p, to_p) in enumerate(cuts):
            ax.plot(
                [from_p.x, to_p.x], [from_p.y, to_p.y], "r-", linewidth=1.5, alpha=0.7
            )
            # Mark cut endpoints
            ax.plot(from_p.x, from_p.y, "ro", markersize=4)
            ax.plot(to_p.x, to_p.y, "ro", markersize=4)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        ax.legend()
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

    def _plot_cut_movements(
        self,
        ax,
        initial_cuts: list[tuple[Point, Point]],
        final_cuts: list[tuple[Point, Point]],
    ):
        """Plot how cuts moved during optimization."""
        # Plot cake outline
        ext_x, ext_y = self.cake.exterior_shape.exterior.xy
        ax.plot(ext_x, ext_y, "k-", linewidth=1, alpha=0.3)

        # Plot cut movements
        for i, (initial_cut, final_cut) in enumerate(zip(initial_cuts, final_cuts)):
            # Initial cut endpoints (blue)
            ax.plot(
                [initial_cut[0].x, initial_cut[1].x],
                [initial_cut[0].y, initial_cut[1].y],
                "b--",
                linewidth=1,
                alpha=0.5,
            )

            # Final cut endpoints (red)
            ax.plot(
                [final_cut[0].x, final_cut[1].x],
                [final_cut[0].y, final_cut[1].y],
                "r-",
                linewidth=1.5,
                alpha=0.7,
            )

            # Arrows showing movement
            for initial_p, final_p in [
                (initial_cut[0], final_cut[0]),
                (initial_cut[1], final_cut[1]),
            ]:
                if (
                    initial_p.distance(final_p) > 0.01
                ):  # Only show if significant movement
                    ax.annotate(
                        "",
                        xy=(final_p.x, final_p.y),
                        xytext=(initial_p.x, initial_p.y),
                        arrowprops=dict(
                            arrowstyle="->", color="green", lw=1.5, alpha=0.6
                        ),
                    )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Cut Movements (Blue→Red)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="b", linestyle="--", label="Initial"),
            Line2D([0], [0], color="r", linestyle="-", label="Final"),
            Line2D(
                [0], [0], color="green", marker=">", linestyle="-", label="Movement"
            ),
        ]
        ax.legend(handles=legend_elements)

    def _plot_piece_distribution(self, ax, cuts: list[tuple[Point, Point]], title: str):
        """Plot distribution of piece sizes."""
        cake_copy = self.cake.copy()
        for from_p, to_p in cuts:
            cake_copy.cut(from_p, to_p)

        pieces = cake_copy.get_pieces()
        areas = sorted([p.area for p in pieces])

        ax.bar(
            range(1, len(areas) + 1),
            areas,
            color="skyblue",
            edgecolor="navy",
            alpha=0.7,
        )
        ax.axhline(
            y=self.cake.get_area() / self.children,
            color="r",
            linestyle="--",
            linewidth=2,
            label="Target Area",
        )
        ax.set_xlabel("Piece #")
        ax.set_ylabel("Area (cm²)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add span annotation
        span = max(areas) - min(areas)
        ax.text(
            0.5,
            0.95,
            f"Span: {span:.4f} cm²",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    def _plot_ratio_comparison(
        self,
        ax,
        initial_cuts: list[tuple[Point, Point]],
        final_cuts: list[tuple[Point, Point]],
    ):
        """Plot comparison of crust ratios."""
        # Get initial ratios
        cake_copy = self.cake.copy()
        for from_p, to_p in initial_cuts:
            cake_copy.cut(from_p, to_p)
        initial_ratios = sorted(cake_copy.get_piece_ratios())

        # Get final ratios
        cake_copy = self.cake.copy()
        for from_p, to_p in final_cuts:
            cake_copy.cut(from_p, to_p)
        final_ratios = sorted(cake_copy.get_piece_ratios())

        x = range(1, len(initial_ratios) + 1)
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            initial_ratios,
            width,
            label="Initial",
            color="lightcoral",
            alpha=0.7,
        )
        ax.bar(
            [i + width / 2 for i in x],
            final_ratios,
            width,
            label="Final",
            color="lightgreen",
            alpha=0.7,
        )

        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area
        ax.axhline(
            y=target_ratio, color="b", linestyle="--", linewidth=2, label="Target Ratio"
        )

        ax.set_xlabel("Piece #")
        ax.set_ylabel("Crust Ratio")
        ax.set_title("Crust Ratio Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add variance annotation
        initial_var = stdev(initial_ratios) if len(initial_ratios) > 1 else 0
        final_var = stdev(final_ratios) if len(final_ratios) > 1 else 0
        ax.text(
            0.5,
            0.95,
            f"Std Dev: {initial_var:.4f} → {final_var:.4f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    def _get_size_span(self) -> float:
        """Helper to get current size span."""
        pieces = self.cake.get_pieces()
        areas = [p.area for p in pieces]
        return max(areas) - min(areas)

    def _get_ratio_variance(self) -> float:
        """Helper to get current ratio variance."""
        ratios = self.cake.get_piece_ratios()
        if len(ratios) > 1:
            return stdev(ratios)
        return 0.0

    def _print_cut_details(self, cuts: list[tuple[Point, Point]]):
        """Helper to print details about a set of cuts."""
        cake_copy = self.cake.copy()
        for from_p, to_p in cuts:
            cake_copy.cut(from_p, to_p)

        pieces = cake_copy.get_pieces()
        areas = [p.area for p in pieces]
        ratios = cake_copy.get_piece_ratios()

        print(f"Pieces: {len(pieces)}")
        print(f"Areas: {[f'{a:.2f}' for a in sorted(areas)]}")
        print(
            f"  Min: {min(areas):.2f}, Max: {max(areas):.2f}, Span: {max(areas) - min(areas):.2f}"
        )
        print(f"Ratios: {[f'{r:.4f}' for r in ratios]}")
        if len(ratios) > 1:
            print(f"  Variance: {stdev(ratios):.6f}")

    def try_divide_and_conquer_cut(
        self, piece: Polygon, num_children: int, target_ratio: float, depth: int = 0
    ) -> tuple[list[tuple[Point, Point]], float] | None:
        """Try to cut a piece for num_children using divide-and-conquer with different ratios.

        Args:
            piece: The polygon piece to divide
            num_children: Number of children to serve from this piece
            target_ratio: Target crust ratio
            depth: Recursion depth for logging

        Returns:
            Tuple of (cuts, score) or None if failed
        """
        if num_children == 1:
            # Base case: no more cuts needed
            return ([], 0.0)

        if num_children == 2:
            # Base case: just split in half (1:1 ratio)
            target_area = piece.area / 2

            best_cut = None
            best_score = float("inf")

            # Try fewer angles for base case
            num_attempts = 10  # Much fewer attempts
            for _ in range(num_attempts):
                angle = random.uniform(0, 180)
                position = self.binary_search(piece, target_area, angle)

                if position is None:
                    continue

                cut_line = self.find_line(position, piece, angle)
                cut_points = self.find_cuts(cut_line, piece)

                if cut_points is None:
                    continue

                from_p, to_p = cut_points

                # Simulate cut and evaluate
                test_pieces = split(piece, cut_line)
                if len(test_pieces.geoms) != 2:
                    continue

                p1, p2 = test_pieces.geoms
                ratio1 = self.cake.get_piece_ratio(p1)
                ratio2 = self.cake.get_piece_ratio(p2)

                # Score based on ratio uniformity and size balance
                ratio_error = abs(ratio1 - ratio2)
                size_error = abs(p1.area - p2.area) / piece.area
                score = ratio_error * 2.0 + size_error * 1.0

                if score < best_score:
                    best_score = score
                    best_cut = (from_p, to_p)

                    # Early stopping if good enough
                    if score < 0.05:
                        break

            if best_cut:
                return ([best_cut], best_score)
            return None

        # Try random (ratio, angle) pairs together
        best_result = None
        best_total_score = float("inf")

        max_ratio_numerator = num_children // 2
        if max_ratio_numerator < 1:
            return None

        # Number of random (ratio, angle) attempts
        # Drastically reduce to prevent exponential blowup
        if depth == 0:
            num_attempts = 10  # Very limited at top level
        elif depth == 1:
            num_attempts = 5  # Even fewer at depth 1
        else:
            num_attempts = 3  # Minimal at deeper levels

        for attempt in range(num_attempts):
            # Randomly select BOTH ratio AND angle together
            split_count = random.randint(1, max_ratio_numerator)
            angle = random.uniform(0, 180)

            # Try to cut off split_count children's worth at this angle
            target_area = piece.area * split_count / num_children
            remaining_count = num_children - split_count

            position = self.binary_search(piece, target_area, angle)
            if position is None:
                continue

            cut_line = self.find_line(position, piece, angle)
            cut_points = self.find_cuts(cut_line, piece)
            if cut_points is None:
                continue

            from_p, to_p = cut_points

            # Simulate the cut
            test_pieces = split(piece, cut_line)
            if len(test_pieces.geoms) != 2:
                continue

            p1, p2 = test_pieces.geoms

            # Identify which piece is for split_count children
            if p1.area < p2.area:
                small_piece, large_piece = p1, p2
                small_count, large_count = split_count, remaining_count
            else:
                small_piece, large_piece = p2, p1
                small_count, large_count = split_count, remaining_count

            # Quick score for this cut
            ratio1 = self.cake.get_piece_ratio(p1)
            ratio2 = self.cake.get_piece_ratio(p2)
            cut_score = abs(ratio1 - target_ratio) + abs(ratio2 - target_ratio)

            # Recursively divide both pieces
            result1 = self.try_divide_and_conquer_cut(
                small_piece, small_count, target_ratio, depth + 1
            )
            result2 = self.try_divide_and_conquer_cut(
                large_piece, large_count, target_ratio, depth + 1
            )

            if result1 is None or result2 is None:
                continue

            cuts1, score1 = result1
            cuts2, score2 = result2

            # Combine results
            all_cuts = [(from_p, to_p)] + cuts1 + cuts2
            total_score = cut_score + score1 + score2

            if total_score < best_total_score:
                best_total_score = total_score
                best_result = (all_cuts, total_score)

                # Early stopping: if we found a very good solution, stop searching
                if total_score < 0.1 * num_children:  # Good enough threshold
                    break

        return best_result

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Main cutting logic - greedy approach with random (ratio, angle) pairs + post-processing"""
        print(f"__________Cutting for {self.children} children_______")

        target_area = self.cake.get_area() / self.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area
        print(f"TARGET AREA: {target_area:.2f} cm²")
        print(f"TARGET CRUST RATIO: {target_ratio:.3f}")
        print("Strategy: Greedy cutting with random ratio+angle exploration\n")

        # Get initial cuts from the greedy algorithm
        initial_cuts = self._greedy_ratio_angle_cutting(target_area, target_ratio)

        # Apply gradient descent post-processing to optimize the cuts
        optimized_cuts = self.gradient_descent_post_processing(initial_cuts)

        return optimized_cuts

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
            num_attempts = self.num_angle_attempts
            cardinal_angles = [0, 90, 180, 270]

            best_cut = None
            best_score = float("inf")
            best_split_ratio = None
            valid_attempts = 0

            # Track best score for each split ratio
            split_ratio_scores = {}
            for split_children in range(min_split, max_split + 1):
                split_ratio_scores[split_children] = float("inf")

            # Build list of (split_ratio, angle) to try
            attempts_to_try = []

            # First: Try all split ratios with all cardinal angles
            for split_children in range(min_split, max_split + 1):
                for angle in cardinal_angles:
                    attempts_to_try.append((split_children, angle, "phase1"))

            # Phase 1: Random sample all split ratios (first half)
            phase1_attempts = num_attempts // 2
            for _ in range(phase1_attempts):
                split_children = random.randint(min_split, max_split)
                angle = random.uniform(0, 180)
                attempts_to_try.append((split_children, angle, "phase1"))

            # Try phase 1 attempts
            for split_children, angle, phase in attempts_to_try:
                remaining_children = cutting_num_children - split_children

                # Calculate target area for this split
                target_cut_area = target_area * split_children

                # Find the cut position using binary search
                position = self.binary_search(cutting_piece, target_cut_area, angle)
                if position is None:
                    continue

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

                # Get crust ratios
                ratio1 = self.cake.get_piece_ratio(small_piece)
                ratio2 = self.cake.get_piece_ratio(large_piece)

                valid_attempts += 1

                # Score this cut
                size_error = abs(small_piece.area - target_cut_area)
                ratio_error = abs(ratio1 - target_ratio) + abs(ratio2 - target_ratio)
                score = size_error * 3.0 + ratio_error * 1.0

                # Track best score for this split ratio
                if score < split_ratio_scores[split_children]:
                    split_ratio_scores[split_children] = score

                if score < best_score:
                    best_score = score
                    best_cut = (
                        from_p,
                        to_p,
                        small_piece,
                        large_piece,
                        ratio1,
                        ratio2,
                        angle,
                    )
                    best_split_ratio = (split_children, remaining_children)

            # Phase 2: Use the best split ratio found, only vary angles
            if split_ratio_scores:
                # Find the split ratio with the best score
                best_ratio_from_phase1 = min(
                    split_ratio_scores.keys(), key=lambda k: split_ratio_scores[k]
                )
                phase2_attempts = num_attempts - phase1_attempts

                print(
                    f"  Phase 1 complete. Best split ratio: {best_ratio_from_phase1}/{cutting_num_children}"
                )
                print(
                    f"  Phase 2: Trying {phase2_attempts} more angles with best ratio..."
                )

                remaining_children_phase2 = (
                    cutting_num_children - best_ratio_from_phase1
                )
                target_cut_area_phase2 = target_area * best_ratio_from_phase1

                # In phase 2, try cardinal angles again with the best ratio, then random
                phase2_angles = list(cardinal_angles) + [
                    random.uniform(0, 180)
                    for _ in range(phase2_attempts - len(cardinal_angles))
                ]

                for angle in phase2_angles:
                    # Only vary angle, keep the best split ratio
                    split_children = best_ratio_from_phase1
                    remaining_children = remaining_children_phase2

                    # Find the cut position using binary search
                    position = self.binary_search(
                        cutting_piece, target_cut_area_phase2, angle
                    )
                    if position is None:
                        continue

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
                    if abs(p1.area - target_cut_area_phase2) < abs(
                        p2.area - target_cut_area_phase2
                    ):
                        small_piece, large_piece = p1, p2
                    else:
                        small_piece, large_piece = p2, p1

                    # Get crust ratios
                    ratio1 = self.cake.get_piece_ratio(small_piece)
                    ratio2 = self.cake.get_piece_ratio(large_piece)

                    valid_attempts += 1

                    # Score this cut
                    size_error = abs(small_piece.area - target_cut_area_phase2)
                    ratio_error = abs(ratio1 - target_ratio) + abs(
                        ratio2 - target_ratio
                    )
                    score = size_error * 3.0 + ratio_error * 1.0

                    if score < best_score:
                        best_score = score
                        best_cut = (
                            from_p,
                            to_p,
                            small_piece,
                            large_piece,
                            ratio1,
                            ratio2,
                            angle,
                        )
                        best_split_ratio = (split_children, remaining_children)

            if best_cut is None:
                print(f"  No valid cut found after {num_attempts} attempts!")
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

    def _sequential_cutting(self) -> list[tuple[Point, Point]]:
        """Fallback: Sequential per-piece cutting with angle selection"""
        target_area = self.cake.get_area() / self.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area

        cuts = []
        cake_copy = self.cake.copy()

        for cut_idx in range(self.children - 1):
            print(f"=== Cut {cut_idx + 1}/{self.children - 1} ===")

            current_pieces = cake_copy.get_pieces()
            cutting_piece = max(current_pieces, key=lambda pc: pc.area)

            print(f"Cutting piece area: {cutting_piece.area:.2f}")

            is_last_cut = cut_idx == self.children - 2

            best_cut = None
            best_score = float("inf")
            best_angle = None

            num_attempts = (
                self.num_angle_attempts * 2 if is_last_cut else self.num_angle_attempts
            )
            valid_attempts = 0

            for attempt in range(num_attempts):
                angle = random.uniform(0, 180)

                result = self.find_best_cut_at_angle(
                    cutting_piece, target_area, angle, cake_copy, is_last_cut
                )

                if result is not None:
                    from_p, to_p, piece_ratio, piece_area = result

                    size_error = abs(piece_area - target_area) / target_area
                    ratio_error = abs(piece_ratio - target_ratio)
                    score = size_error * 3.0 + ratio_error * 1.0

                    valid_attempts += 1

                    if score < best_score:
                        best_score = score
                        best_cut = (from_p, to_p)
                        best_angle = angle
                        best_ratio = piece_ratio
                        best_size = piece_area

            if best_cut is None:
                print("  Failed: No valid cut found")
                continue

            from_p, to_p = best_cut
            try:
                cake_copy.cut(from_p, to_p)
                cuts.append((from_p, to_p))

                areas = [p.area for p in cake_copy.get_pieces()]
                size_error = abs(best_size - target_area)
                ratio_error = abs(best_ratio - target_ratio)
                print(
                    f"  Best angle: {best_angle:.1f}° (tried {valid_attempts} valid angles)"
                )
                print(
                    f"  Piece: size={best_size:.2f} (target={target_area:.2f}, err={size_error:.2f}), ratio={best_ratio:.3f} (target={target_ratio:.3f}, err={ratio_error:.3f})"
                )
                print(f"  Current areas: {[f'{a:.2f}' for a in sorted(areas)]}\n")

            except Exception as e:
                print(f"  Error applying cut: {e}")
                break

        return cuts
