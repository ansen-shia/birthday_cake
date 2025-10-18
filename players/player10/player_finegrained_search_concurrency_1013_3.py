from shapely.geometry import LineString, Point, Polygon
import math
import random
from statistics import stdev
from concurrent.futures import ProcessPoolExecutor, as_completed

from players.player import Player
from src.cake import Cake
from shapely.ops import split

COMPUTATION_RATIO = 1
PHRASE_ONE_TOTAL_ATTEMPS = 60 * 9 * COMPUTATION_RATIO
PHRASE_TWO_TOTAL_ATTEMPS = 360 * 9 * COMPUTATION_RATIO
PHRASE_THREE_TOTAL_ATTEMPS = 90 * 9 * COMPUTATION_RATIO
PHRASE_THREE_STEP = 2


class Player10(Player):
    def __init__(
        self,
        children: int,
        cake: Cake,
        cake_path: str | None,
        # phrase_three_attempts: int = 180,
        num_of_processes: int = 8,
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
        # But clip it to stay within the piece boundaries
        dx = math.cos(angle_rad) * max_dim
        dy = math.sin(angle_rad) * max_dim

        # Find intersection points with the piece boundary
        test_line = LineString(
            [(point_x - dx, point_y - dy), (point_x + dx, point_y + dy)]
        )
        intersections = test_line.intersection(piece.boundary)

        if intersections.is_empty:
            return None

        # Get the intersection points
        if intersections.geom_type == "Point":
            points = [intersections]
        elif intersections.geom_type == "MultiPoint":
            points = list(intersections.geoms)
        else:
            return None

        if len(points) >= 2:
            # Use the two intersection points as the cut endpoints
            cut_line = LineString(points[:2])
            return cut_line

        return None

    def find_cuts(self, line: LineString, piece: Polygon):
        """Find exactly two points where the cut line intersects the cake boundary."""
        intersection = line.intersection(piece.boundary)

        # Collect all intersection points
        points = []
        if intersection.is_empty:
            return None
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

        # Remove duplicates
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

        if len(unique_points) < 2:
            return None

        # If exactly 2 points, use them
        if len(unique_points) == 2:
            return (unique_points[0], unique_points[1])

        # For complex shapes, try consecutive pairs first
        for i in range(len(unique_points) - 1):
            p1, p2 = unique_points[i], unique_points[i + 1]
            test_line = LineString([p1, p2])
            midpoint = test_line.interpolate(0.5, normalized=True)
            if piece.contains(midpoint) or piece.boundary.contains(midpoint):
                return (p1, p2)

        # If no consecutive pairs work, try all pairs
        from itertools import combinations

        for p1, p2 in combinations(unique_points, 2):
            test_line = LineString([p1, p2])
            midpoint = test_line.interpolate(0.5, normalized=True)
            if piece.contains(midpoint) or piece.boundary.contains(midpoint):
                return (p1, p2)

        return None

    def _is_valid_cut_pair(
        self, p1: Point, p2: Point, piece: Polygon, original_line: LineString
    ) -> bool:
        """Check if a pair of points forms a valid cut for the piece."""
        test_line = LineString([p1, p2])

        # Strategy 1: Check if midpoint is inside the piece (most important)
        midpoint = test_line.interpolate(0.5, normalized=True)
        if not (piece.contains(midpoint) or piece.boundary.contains(midpoint)):
            return False

        # Strategy 2: Verify this cut would split into exactly 2 pieces
        from shapely.ops import split as shapely_split

        try:
            result = shapely_split(piece, test_line)
            if len(result.geoms) != 2:
                return False
        except Exception:
            # If splitting fails, this is likely not a valid cut
            return False

        # Strategy 3: Additional validation - check if the cut line is reasonable
        # The cut should not be too long compared to the piece size
        cut_length = p1.distance(p2)
        piece_width = piece.bounds[2] - piece.bounds[0]
        if cut_length > piece_width * 1.5:  # Too long, likely invalid
            return False

        return True

    def _is_valid_cut_pair_conservative(
        self, p1: Point, p2: Point, piece: Polygon, original_line: LineString
    ) -> bool:
        """Conservative validation that prioritizes simple, clean cuts for complex shapes."""
        test_line = LineString([p1, p2])

        # Strategy 1: Check if midpoint is inside the piece (most important)
        midpoint = test_line.interpolate(0.5, normalized=True)
        if not (piece.contains(midpoint) or piece.boundary.contains(midpoint)):
            return False

        # Strategy 2: Verify this cut would split into exactly 2 pieces
        from shapely.ops import split as shapely_split

        try:
            result = shapely_split(piece, test_line)
            if len(result.geoms) != 2:
                return False
        except Exception:
            # If splitting fails, this is likely not a valid cut
            return False

        # Strategy 3: Conservative size validation - avoid very small pieces
        piece1, piece2 = result.geoms
        min_area = min(piece1.area, piece2.area)
        total_area = piece.area

        # Each piece should be at least 10% of total area (more conservative than before)
        if min_area < total_area * 0.1:
            return False

        # Strategy 4: Conservative aspect ratio check - avoid very elongated pieces
        for geom in result.geoms:
            if geom.area > 0:
                bounds = geom.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                aspect_ratio = (
                    max(width, height) / min(width, height)
                    if min(width, height) > 0
                    else float("inf")
                )

                # Be more conservative - reject very elongated pieces
                if aspect_ratio > 8:  # More conservative than before
                    return False

        # Strategy 5: Check for clean cuts - avoid cuts that intersect boundary multiple times
        # This is especially important for Hilbert-like curves
        extended_test = test_line.buffer(0.05)  # Smaller buffer for cleaner cuts
        boundary_intersections = extended_test.intersection(piece.boundary)

        if hasattr(boundary_intersections, "geoms"):
            intersection_count = len(boundary_intersections.geoms)
        else:
            intersection_count = 1 if not boundary_intersections.is_empty else 0

        # Reject cuts that intersect boundary too many times (indicates complex intersections)
        if intersection_count > 2:
            return False

        return True

    def _find_cuts_for_complex_shape(
        self, line: LineString, piece: Polygon
    ) -> tuple[Point, Point] | None:
        """Specialized method for complex concave shapes like Hilbert curves."""
        intersection = line.intersection(piece.boundary)

        # For very complex shapes, use a more adaptive approach
        if intersection.is_empty:
            return None

        # Collect intersection points
        points = []
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

        # Remove duplicates
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

        if len(unique_points) < 2:
            return None

        # For complex shapes, try a different strategy:
        # Instead of trying all pairs, use a more intelligent approach

        # Strategy 1: Try to find cuts that are roughly perpendicular to the piece's orientation
        # Calculate the piece's bounding box orientation
        bounds = piece.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # If the piece is wider than tall, prefer horizontal-ish cuts
        if width > height * 1.2:
            # Try horizontal cuts first (angles around 0° and 180°)
            horizontal_angles = [0, 180, 45, 135, 90, 270]
            for angle in horizontal_angles:
                test_line = self.find_line(0.5, piece, angle)  # Use center position
                result = self.find_cuts(test_line, piece)
                if result:
                    return result
        else:
            # Try vertical cuts first (angles around 90° and 270°)
            vertical_angles = [90, 270, 45, 135, 0, 180]
            for angle in vertical_angles:
                test_line = self.find_line(0.5, piece, angle)
                result = self.find_cuts(test_line, piece)
                if result:
                    return result

        # Strategy 2: If orientation-based approach fails, fall back to simple logic
        # Use basic intersection logic without calling find_cuts recursively
        intersection = line.intersection(piece.boundary)

        points = []
        if not intersection.is_empty:
            if intersection.geom_type == "Point":
                points = [intersection]
            elif intersection.geom_type == "MultiPoint":
                points = list(intersection.geoms)
            elif intersection.geom_type == "LineString":
                coords = list(intersection.coords)
                points = [Point(coords[0]), Point(coords[-1])]

        if len(points) >= 2:
            # Simple approach: return the first two points
            return (points[0], points[1])

        return None

    def _is_complex_shape(self, piece: Polygon) -> bool:
        """Determine if a piece is a complex shape that needs special handling."""
        # Enhanced heuristics to detect complex shapes like Hilbert curves

        # 1. Check the number of vertices - complex shapes tend to have more vertices
        if hasattr(piece, "exterior") and piece.exterior:
            num_vertices = (
                len(piece.exterior.coords) - 1
            )  # -1 because first and last are the same
            if num_vertices > 15:  # Lower threshold for Hilbert curves
                return True

        # 2. Check aspect ratio - very elongated or square-like shapes
        bounds = piece.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        aspect_ratio = (
            max(width, height) / min(width, height)
            if min(width, height) > 0
            else float("inf")
        )

        # Very elongated or very square shapes might be complex
        if aspect_ratio > 3 or aspect_ratio < 1.2:
            return True

        # 3. Check boundary complexity - Hilbert curves have very complex boundaries
        if hasattr(piece, "boundary") and piece.boundary:
            boundary_length = piece.boundary.length
            area = piece.area

            # Complex shapes tend to have higher perimeter-to-area ratio
            if area > 0:
                complexity_ratio = boundary_length / area
                if complexity_ratio > 1.8:  # Lower threshold for Hilbert curves
                    return True

        # 4. Check if the shape is very small compared to its bounding box (irregular shape)
        if area > 0:
            bbox_area = width * height
            fill_ratio = area / bbox_area
            if fill_ratio < 0.4:  # More lenient for complex shapes
                return True

        # 5. Special check for known complex patterns
        # Hilbert curves often have characteristic patterns
        if hasattr(piece, "exterior") and piece.exterior:
            coords = list(piece.exterior.coords)
            if len(coords) > 10:
                # Check for repeating patterns or fractal-like structures
                # This is a simple heuristic - in practice, more sophisticated analysis could be done
                x_coords = [c[0] for c in coords[:-1]]  # Exclude closing point
                y_coords = [c[1] for c in coords[:-1]]

                # Check for regular spacing patterns (characteristic of Hilbert curves)
                x_unique = sorted(set(x_coords))
                y_unique = sorted(set(y_coords))

                # If coordinates are very regularly spaced, it might be a Hilbert curve
                if len(x_unique) > 5 and len(y_unique) > 5:
                    x_spacing = [
                        x_unique[i + 1] - x_unique[i] for i in range(len(x_unique) - 1)
                    ]
                    y_spacing = [
                        y_unique[i + 1] - y_unique[i] for i in range(len(y_unique) - 1)
                    ]

                    # Check if spacings are very regular (Hilbert curve characteristic)
                    x_variance = sum(
                        (s - sum(x_spacing) / len(x_spacing)) ** 2 for s in x_spacing
                    ) / len(x_spacing)
                    y_variance = sum(
                        (s - sum(y_spacing) / len(y_spacing)) ** 2 for s in y_spacing
                    ) / len(y_spacing)

                    if x_variance < 0.1 and y_variance < 0.1:  # Very regular spacing
                        return True

        return False

    def _is_valid_cut_pair_robust(
        self, p1: Point, p2: Point, piece: Polygon, original_line: LineString
    ) -> bool:
        """Robust validation combining multiple strategies for complex concave shapes."""
        test_line = LineString([p1, p2])

        # Strategy 1: Midpoint validation (essential)
        midpoint = test_line.interpolate(0.5, normalized=True)
        if not (piece.contains(midpoint) or piece.boundary.contains(midpoint)):
            return False

        # Strategy 2: Split validation (essential)
        from shapely.ops import split as shapely_split

        try:
            result = shapely_split(piece, test_line)
            if len(result.geoms) != 2:
                return False
        except Exception:
            return False

        # Strategy 3: Size validation (prevent tiny fragments)
        piece1, piece2 = result.geoms
        min_area = min(piece1.area, piece2.area)
        total_area = piece.area

        if min_area < total_area * 0.05:  # More lenient for complex shapes
            return False

        # Strategy 4: Aspect ratio validation (prevent very elongated pieces)
        for geom in result.geoms:
            if geom.area > 0:
                bounds = geom.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                aspect_ratio = (
                    max(width, height) / min(width, height)
                    if min(width, height) > 0
                    else float("inf")
                )

                if aspect_ratio > 12:  # More lenient for complex shapes
                    return False

        # Strategy 5: Boundary intersection analysis (for Hilbert-like curves)
        # Check if cut intersects boundary cleanly
        extended_test = test_line.buffer(0.02)  # Very small buffer
        boundary_intersections = extended_test.intersection(piece.boundary)

        if hasattr(boundary_intersections, "geoms"):
            intersection_count = len(boundary_intersections.geoms)
            # For complex shapes, allow up to 4 intersection points if they're close
            if intersection_count > 4:
                return False
        else:
            intersection_count = 1 if not boundary_intersections.is_empty else 0

        # Strategy 6: Cut length validation (avoid extremely long cuts)
        cut_length = p1.distance(p2)
        piece_width = piece.bounds[2] - piece.bounds[0]
        if cut_length > piece_width * 2.0:  # Allow longer cuts for complex shapes
            return False

        return True

    def _is_valid_cut_pair_enhanced(
        self, p1: Point, p2: Point, piece: Polygon, original_line: LineString
    ) -> bool:
        """Enhanced validation for complex concave shapes like Hilbert curves."""
        test_line = LineString([p1, p2])

        # Strategy 1: Check if midpoint is inside the piece (most important)
        midpoint = test_line.interpolate(0.5, normalized=True)
        if not (piece.contains(midpoint) or piece.boundary.contains(midpoint)):
            return False

        # Strategy 2: Verify this cut would split into exactly 2 pieces
        from shapely.ops import split as shapely_split

        try:
            result = shapely_split(piece, test_line)
            if len(result.geoms) != 2:
                return False
        except Exception:
            # If splitting fails, this is likely not a valid cut
            return False

        # Strategy 3: Enhanced validation for complex shapes
        # Check that both resulting pieces have reasonable areas (not tiny fragments)
        piece1, piece2 = result.geoms
        min_area = min(piece1.area, piece2.area)
        total_area = piece.area

        # Each piece should be at least 5% of total area (avoid tiny fragments)
        if min_area < total_area * 0.05:
            return False

        # Strategy 4: Check that the cut line doesn't create pieces that are too elongated
        # (elongated pieces are harder to cut fairly)
        for geom in result.geoms:
            if geom.area > 0:
                bounds = geom.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                aspect_ratio = (
                    max(width, height) / min(width, height)
                    if min(width, height) > 0
                    else float("inf")
                )

                # If aspect ratio is too extreme, this might not be a good cut
                if aspect_ratio > 10:  # Very elongated piece
                    return False

        # Strategy 5: For Hilbert-like curves, prefer cuts that don't create too many "branches"
        # Check if the cut intersects the boundary only at the endpoints (clean cut)
        extended_test = test_line.buffer(
            0.1
        )  # Small buffer to catch nearby intersections
        boundary_intersections = extended_test.intersection(piece.boundary)

        # Count distinct intersection regions (should be minimal for clean cuts)
        if hasattr(boundary_intersections, "geoms"):
            intersection_count = len(boundary_intersections.geoms)
        else:
            intersection_count = 1 if not boundary_intersections.is_empty else 0

        # Too many intersection regions suggest a messy cut
        if intersection_count > 3:
            return False

        return True

    def calculate_piece_area(self, piece: Polygon, position: float, angle: float):
        """Determines the area of the pieces we cut.

        Args:
            piece: The polygon piece to cut
            position: Position along sweep direction (0 to 1)
            angle: Angle in degrees for the cutting line
        """
        line = self.find_line(position, piece, angle)
        if line is None:
            return 0.0
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
                f"  Phase 1: Processing {len(attempts_to_try)} attempts concurrently with {self.num_of_processes} processes..."
            )

            # Split attempts into chunks for each process
            chunk_size = max(1, len(attempts_to_try) // self.num_of_processes)
            attempt_chunks = [
                attempts_to_try[i : i + chunk_size]
                for i in range(0, len(attempts_to_try), chunk_size)
            ]

            # Process chunks concurrently
            with ProcessPoolExecutor(max_workers=self.num_of_processes) as executor:
                # Submit all chunks
                futures = []
                for chunk in attempt_chunks:
                    future = executor.submit(
                        self._process_batch,
                        chunk,
                        cutting_piece,
                        cutting_num_children,
                        target_area,
                        target_ratio,
                    )
                    futures.append(future)

                # Collect results as they complete
                for future in as_completed(futures):
                    batch_result = future.result()
                    if batch_result:
                        valid_attempts += 1

                        # Update best scores for this batch
                        split_children = batch_result["split_children"]
                        if batch_result["score"] < split_ratio_scores[split_children]:
                            split_ratio_scores[split_children] = batch_result["score"]

                        if batch_result["score"] < best_score:
                            best_score = batch_result["score"]
                            best_cut = (
                                batch_result["cut_points"][0],
                                batch_result["cut_points"][1],
                                batch_result["small_piece"],
                                batch_result["large_piece"],
                                batch_result["ratio1"],
                                batch_result["ratio2"],
                                batch_result["angle"],
                            )
                            best_split_ratio = (
                                split_children,
                                batch_result["remaining_children"],
                            )

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
                    f"  Phase 2: Trying {self.phrase_two_attempts} more angles with best ratio..."
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
                print(
                    f"  Phase 2: Processing {len(phase2_attempts_to_try)} attempts concurrently with {self.num_of_processes} processes..."
                )

                # Split Phase 2 attempts into chunks for each process
                chunk_size_phase2 = max(
                    1, len(phase2_attempts_to_try) // self.num_of_processes
                )
                phase2_chunks = [
                    phase2_attempts_to_try[i : i + chunk_size_phase2]
                    for i in range(0, len(phase2_attempts_to_try), chunk_size_phase2)
                ]

                # Process Phase 2 chunks concurrently
                with ProcessPoolExecutor(max_workers=self.num_of_processes) as executor:
                    # Submit all chunks
                    futures = []
                    for chunk in phase2_chunks:
                        future = executor.submit(
                            self._process_batch,
                            chunk,
                            cutting_piece,
                            cutting_num_children,
                            target_area,
                            target_ratio,
                        )
                        futures.append(future)

                    # Collect results as they complete
                    for future in as_completed(futures):
                        batch_result = future.result()
                        if batch_result:
                            valid_attempts += 1

                            if batch_result["score"] < best_score:
                                best_score = batch_result["score"]
                                best_cut = (
                                    batch_result["cut_points"][0],
                                    batch_result["cut_points"][1],
                                    batch_result["small_piece"],
                                    batch_result["large_piece"],
                                    batch_result["ratio1"],
                                    batch_result["ratio2"],
                                    batch_result["angle"],
                                )
                                best_split_ratio = (
                                    batch_result["split_children"],
                                    batch_result["remaining_children"],
                                )

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

                phase3_attempts_to_try = [
                    (best_ratio_from_phase1, angle, "phase3") for angle in phase3_angles
                ]

                # Process Phase 3 attempts concurrently
                print(
                    f"  Phase 3: Processing {len(phase3_attempts_to_try)} attempts concurrently with {self.num_of_processes} processes..."
                )

                # Split Phase 3 attempts into chunks for each process
                chunk_size_phase3 = max(
                    1, len(phase3_attempts_to_try) // self.num_of_processes
                )
                phase3_chunks = [
                    phase3_attempts_to_try[i : i + chunk_size_phase3]
                    for i in range(0, len(phase3_attempts_to_try), chunk_size_phase3)
                ]

                # Process Phase 3 chunks concurrently
                with ProcessPoolExecutor(max_workers=self.num_of_processes) as executor:
                    # Submit all chunks
                    futures = []
                    for chunk in phase3_chunks:
                        future = executor.submit(
                            self._process_batch,
                            chunk,
                            cutting_piece,
                            cutting_num_children,
                            target_area,
                            target_ratio,
                        )
                        futures.append(future)

                    # Collect results as they complete
                    for future in as_completed(futures):
                        batch_result = future.result()
                        if batch_result:
                            valid_attempts += 1

                            if batch_result["score"] < best_score:
                                best_score = batch_result["score"]
                                best_cut = (
                                    batch_result["cut_points"][0],
                                    batch_result["cut_points"][1],
                                    batch_result["small_piece"],
                                    batch_result["large_piece"],
                                    batch_result["ratio1"],
                                    batch_result["ratio2"],
                                    batch_result["angle"],
                                )
                                best_split_ratio = (
                                    batch_result["split_children"],
                                    batch_result["remaining_children"],
                                )

                print(
                    f"    Phase 3 complete. Final best angle: {best_cut[6]:.1f}° with score {best_score:.3f}"
                )
            else:
                print(f"    Best cut found with score {best_score:.3f}")
            if best_cut is None:
                print(
                    f"  No valid cut found after {len(attempts_to_try) + len(phase2_attempts_to_try) + self.phrase_three_attempts} attempts!"
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

        # Additional refinement for complex shapes to improve crust ratio consistency
        # Check if the original cake shape is complex (before any cuts)
        original_shape = cake_copy.exterior_shape
        if self._is_complex_shape(original_shape):
            print("Applying additional refinement for complex shape...")
            all_cuts = self._refine_cuts_for_complex_shape(all_cuts, cake_copy)

        return all_cuts

    def _refine_cuts_for_complex_shape(self, cuts: list, cake: Cake) -> list:
        """Apply additional refinement to improve crust ratio consistency for complex shapes."""
        print("  Refining cuts for complex shape...")

        # Strategy 1: Try different cut orders
        if len(cuts) > 2:  # Only for cakes with multiple cuts
            alternative_cuts = self._try_alternative_cut_order(cuts, cake.copy())
            if alternative_cuts:
                # Check if the alternative order gives better crust ratio consistency
                original_ratios = cake.get_piece_ratios()
                alt_cake = cake.copy()
                for cut in alternative_cuts:
                    alt_cake.cut(cut[0], cut[1])

                alt_ratios = alt_cake.get_piece_ratios()

                if len(alt_ratios) > 1 and len(original_ratios) > 1:
                    from statistics import stdev

                    original_stdev = (
                        stdev(original_ratios) if len(original_ratios) > 1 else 0
                    )
                    alt_stdev = stdev(alt_ratios) if len(alt_ratios) > 1 else 0

                    # Use the better (lower stdev) cut order
                    if alt_stdev < original_stdev:
                        print(
                            f"  Alternative cut order improved stdev from {original_stdev:.3f} to {alt_stdev:.3f}"
                        )
                        return alternative_cuts

        # Strategy 2: For very complex shapes, try to optimize individual cuts
        # This is more expensive but can help with stubborn cases
        if len(cuts) > 5:  # Only for very complex cakes
            print("  Attempting individual cut optimization...")
            optimized_cuts = self._optimize_individual_cuts(cuts, cake.copy())
            if optimized_cuts:
                return optimized_cuts

        return cuts

    def _try_alternative_cut_order(self, cuts: list, cake: Cake) -> list | None:
        """Try alternative cut orders for better crust ratio consistency."""
        # For complex shapes, the order of cuts can affect the final crust ratio distribution
        # Try reversing the cut order or other permutations

        if len(cuts) <= 2:
            return None  # Not enough cuts to reorder meaningfully

        # Try reversing the order
        reversed_cuts = list(reversed(cuts))
        test_cake = cake.copy()

        try:
            for cut in reversed_cuts:
                test_cake.cut(cut[0], cut[1])

            # If successful, return the reversed order
            return reversed_cuts
        except:  # noqa: E722
            return None  # Reversed order didn't work

    def _optimize_individual_cuts(self, cuts: list, cake: Cake) -> list | None:
        """Try to optimize individual cuts for better overall consistency."""
        # For very complex shapes, try to adjust problematic cuts
        # This is a simplified approach - in practice, this could be much more sophisticated

        optimized_cuts = []
        cake_copy = cake.copy()

        for i, cut in enumerate(cuts):
            # Apply the cut
            try:
                cake_copy.cut(cut[0], cut[1])
                optimized_cuts.append(cut)
            except:  # noqa: E722
                # If this cut fails, try to find a replacement
                print(f"    Cut {i + 1} failed, trying to find replacement...")
                # For now, just skip this cut (simplified approach)
                # In a more sophisticated implementation, we could try alternative cuts
                continue

        if len(optimized_cuts) == len(cuts):
            return optimized_cuts
        else:
            return None  # Optimization failed
