import os
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split

from players.player import Player
from src.cake import Cake
from . import utils
from players.player4.mushroom import get_mushroom_cuts
from players.player4.rocket import get_rocket_cuts
from players.player4.hilbert import get_hilbert_cuts

import cProfile
import math
import pstats
import time

N_SWEEP_DIRECTIONS = 72
SWEEP_STEP_DISTANCE = 0.1  # cm
N_MAX_SWEEPS_PER_DIR = 100
N_BEST_CUT_CANDIDATES_TO_TRY = 10000


class Player4(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.PRESET_CAKES = [("mushroom", 6), ("rocket", 6), ("hilbert", 31)]

        self.N_SWEEP_DIRECTIONS = N_SWEEP_DIRECTIONS

        target_info = utils.get_final_piece_targets(cake, children)

        self.final_piece_min_area = target_info["min_area"]
        self.final_piece_max_area = target_info["max_area"]
        print(
            f"Player 4: Min area: {self.final_piece_min_area}, max area: {self.final_piece_max_area}."
        )
        self.final_piece_target_ratio = target_info["target_ratio"]
        self.final_piece_min_ratio = target_info["min_ratio"]
        self.final_piece_max_ratio = target_info["max_ratio"]
        print(f"Player 4: Target ratio: {self.final_piece_target_ratio}.")

        self.profiler = cProfile.Profile()

        # When we accept a candidate final piece, if we search strictly for pieces larger than the requirement, we will fail to find a solution since the remaining area is smaller than required. To avoid this, we introduce a small tolerance to the minimum area required for each final piece. Practically, the minimum final piece size that we can accept for a candidate solution is related to the total number of children:
        # Let the target final size (determined by the number of children) be A, and our tolerance be e. Since our total size span tolerance is 0.5, if each candidate final piece other than the last one has area A - e, the last piece will have area A_n = A + (n - 1)e.
        # This means that the maximum size span, when all pieces are as small as possible except the last one, is:
        # size_span = A_n - (A - e) = (A + (n - 1)e) - (A - e) = ne
        # The practical goal here is to keep the size span less than 0.5, so we want ne < 0.5, or e < 0.5/n.
        self.final_piece_target_area = cake.exterior_shape.area / float(children)
        self.final_piece_size_epsilon = 0.5 / float(children)
        self.final_piece_min_area = (
            self.final_piece_target_area - self.final_piece_size_epsilon
        )
        self.final_piece_max_area = (
            self.final_piece_target_area + self.final_piece_size_epsilon
        )
        print(
            f"Player 4: Final piece target area: {self.final_piece_target_area}, epsilon min area: {self.final_piece_min_area}."
        )

        # Similarly, ratio target is 5%, meaning we can accept pieces with ratio +- 2.5%.
        self.final_piece_target_ratio = cake.get_piece_ratio(cake.exterior_shape)
        self.final_piece_min_ratio = self.final_piece_target_ratio - 0.025
        self.final_piece_max_ratio = self.final_piece_target_ratio + 0.025
        print(
            f"Player 4: Final piece target ratio: {self.final_piece_target_ratio}, min ratio: {self.final_piece_min_ratio}, max ratio: {self.final_piece_max_ratio}."
        )

    def get_cuts(self) -> list[tuple[Point, Point]]:
        # If cake is a preset cake, use prepared cuts
        if any(
            p in self.cake_path and self.children == n_children
            for p, n_children in self.PRESET_CAKES
        ):
            return self._get_preset_cuts(os.path.basename(self.cake_path))

        # Otherwise, use DFS search
        print(f"Player 4: Starting search for {self.children} children...")
        self.start_time = time.time()
        self.profiler.enable()
        result = self.DFS(self.cake.exterior_shape, self.children)
        self.end_time = time.time()
        print(
            f"Player 4: Search complete, took {self.end_time - self.start_time} seconds."
        )
        while result is None and self.end_time - self.start_time < 300:
            self.N_SWEEP_DIRECTIONS = self.N_SWEEP_DIRECTIONS * 2
            print(
                f"Increasing sweep directions to {self.N_SWEEP_DIRECTIONS} and retrying..."
            )
            result = self.DFS(self.cake.exterior_shape, self.children)
            self.end_time = time.time()
        if result is None:
            print("Player 4: No solution found.")
            result = []

        self.profiler.disable()
        stats = pstats.Stats(self.profiler)
        stats.sort_stats("cumulative")
        stats.print_stats(10)
        return result

    def _get_preset_cuts(self, cake_path: str) -> list[tuple[Point, Point]]:
        print(f"Using preset cuts for {cake_path}.")
        if "mushroom" in cake_path and self.children == 6:
            return get_mushroom_cuts(self.children, self.cake)
        elif "rocket" in cake_path and self.children == 6:
            return get_rocket_cuts(self.cake)
        elif "hilbert" in cake_path and self.children == 31:
            return get_hilbert_cuts()
        return None

    def DFS(
        self, piece: Polygon, children: int, depth: int = 0
    ) -> list[tuple[Point, Point]]:
        print(f"\nDFS solving for {children} children remaining...")
        if children == 1:
            return []

        piece_is_convex = utils.piece_is_convex(piece)
        if piece_is_convex:
            print("Piece is convex, using optimized cut generation.")
            cuts = self.generate_and_filter_cuts_convex(piece, children)
        else:
            print("Piece is not convex, using standard cut generation.")
            cuts = self.generate_and_filter_cuts(piece, children)

        cut_sequence: list[tuple[Point, Point]] = []
        cuts_tried = 0
        for cut in cuts:
            cuts_tried += 1
            if cuts_tried > N_BEST_CUT_CANDIDATES_TO_TRY:
                print(f"Tried {N_BEST_CUT_CANDIDATES_TO_TRY} cuts, no solution found.")
                break

            from_p, to_p = cut
            result = utils.make_cut(self.cake, piece, from_p, to_p)
            if result is None:
                continue
            subpiece1, subpiece2 = result

            # Central DFS idea: assign each subpiece to have (k, children - k) final pieces and try
            for k in range(1, children):
                subpiece1_children = k
                subpiece2_children = children - k
                subpiece1_min_area = self.final_piece_min_area * subpiece1_children
                subpiece2_min_area = self.final_piece_min_area * subpiece2_children

                # Is each subpiece big enough to share with its assigned children?
                if (
                    subpiece1.area < subpiece1_min_area
                    or subpiece2.area < subpiece2_min_area
                ):
                    continue
                # Is each subpiece too big to share with its assigned children?
                if (
                    subpiece1.area > self.final_piece_max_area * subpiece1_children
                    or subpiece2.area > self.final_piece_max_area * subpiece2_children
                ):
                    continue

                # DFS on each subpiece
                cuts1 = []
                if subpiece1_children > 1:
                    cuts1 = self.DFS(subpiece1, subpiece1_children, depth + 1)
                    if cuts1 is None:
                        continue
                cuts2 = []
                if subpiece2_children > 1:
                    cuts2 = self.DFS(subpiece2, subpiece2_children, depth + 1)
                    if cuts2 is None:
                        continue

                # Success! Found k where both subpieces can be cut properly
                cut_sequence = [(from_p, to_p)] + cuts1 + cuts2
                print(f"{'\t' * depth}Found solution for {children} children.")
                return cut_sequence
        print(f"{'\t' * depth}No solution found for {children} children remaining.")
        return None

    def generate_and_filter_cuts(self, piece: Polygon, children: int) -> list:
        """
        Sweeps from centroid outward for each angle.
        """
        # Collect all boundary coordinates once
        coords = list(piece.exterior.coords)
        if piece.interiors:
            for interior in piece.interiors:
                coords.extend(list(interior.coords))

        # For each angle, generate all cuts for that angle, then sort and filter
        candidate_cuts = []
        cut_angles = []
        degrees_per_cut = 180 / self.N_SWEEP_DIRECTIONS
        for i in range(self.N_SWEEP_DIRECTIONS):
            cut_angles.append(i * degrees_per_cut)
        for angle_deg in cut_angles:
            angle_rad = math.radians(angle_deg)

            # Normal vector (direction we sweep along)
            normal_x = math.cos(angle_rad)
            normal_y = math.sin(angle_rad)

            # Line direction (perpendicular to normal)
            line_dx = -math.sin(angle_rad)
            line_dy = math.cos(angle_rad)

            # Project all boundary points onto the normal axis to find sweep range
            projections = [x * normal_x + y * normal_y for x, y in coords]
            min_proj = min(projections)
            max_proj = max(projections)

            # Project centroid onto normal to get center point
            centroid_proj = piece.centroid.x * normal_x + piece.centroid.y * normal_y
            # Calculate step size
            max_distance = max(
                abs(max_proj - centroid_proj), abs(min_proj - centroid_proj)
            )
            num_sweeps = max(1, int(max_distance / SWEEP_STEP_DISTANCE) * 2 + 1)
            step_dist = SWEEP_STEP_DISTANCE
            if num_sweeps > N_MAX_SWEEPS_PER_DIR:
                num_sweeps = N_MAX_SWEEPS_PER_DIR
                step_dist = max_distance / (num_sweeps // 2)
                # print(f"Piece too big, limiting sweeps to {N_MAX_SWEEPS_PER_DIR}.")

            min_ratio_err_found = float("inf")
            for i in range(num_sweeps):
                if i == 0:
                    offset = 0
                elif i % 2 == 1:  # odd indices go positive
                    offset = ((i + 1) // 2) * step_dist
                else:  # even indices go negative
                    offset = -(i // 2) * step_dist

                proj = centroid_proj + offset

                # Skip if we've gone beyond the bounds
                if proj < min_proj or proj > max_proj:
                    continue

                # Create a line at this projection
                center_x = proj * normal_x
                center_y = proj * normal_y

                # Extend the line far enough to cover the polygon
                extension = 1000
                p1 = (center_x - extension * line_dx, center_y - extension * line_dy)
                p2 = (center_x + extension * line_dx, center_y + extension * line_dy)

                sweep_line = LineString([p1, p2])
                intersection = sweep_line.intersection(piece.boundary)

                if intersection.is_empty:
                    continue

                # Extract points from the intersection
                points: list[Point] = []
                if intersection.geom_type == "Point":
                    points = [intersection]
                elif intersection.geom_type == "MultiPoint":
                    points = list(intersection.geoms)
                elif intersection.geom_type == "GeometryCollection":
                    points = [
                        geom for geom in intersection.geoms if geom.geom_type == "Point"
                    ]

                # Check all pairs of intersection points
                if len(points) >= 2:
                    for j in range(len(points)):
                        for k in range(j + 1, len(points)):
                            from_p: Point = points[j]
                            to_p: Point = points[k]
                            line = LineString([from_p, to_p])

                            line_intersection = line.intersection(piece)
                            if line_intersection.geom_type != "LineString":
                                continue
                            if not line_intersection.equals(line):
                                continue

                            try:
                                split_result = split(piece, line)
                                if len(split_result.geoms) != 2:
                                    continue

                                piece1, piece2 = split_result.geoms
                                piece1_area = piece1.area
                                piece2_area = piece2.area
                                piece1_ratio = self.cake.get_piece_ratio(piece1)
                                piece2_ratio = self.cake.get_piece_ratio(piece2)

                                # Apply filters
                                if (
                                    piece1_area < self.final_piece_min_area
                                    or piece2_area < self.final_piece_min_area
                                ):
                                    continue

                                # Calculate score for sorting
                                ratio1_error = abs(
                                    piece1_ratio - self.final_piece_target_ratio
                                )
                                ratio2_error = abs(
                                    piece2_ratio - self.final_piece_target_ratio
                                )
                                ratio_error = ratio1_error + ratio2_error

                                candidate_cuts.append(((from_p, to_p), ratio_error))
                                if ratio_error < min_ratio_err_found:
                                    min_ratio_err_found = ratio_error

                            except Exception:
                                continue

                # If we found a very good cut, we can stop early for this angle
                if min_ratio_err_found < 0.0125:
                    # print(f"Early stopping sweep at angle {angle_deg} due to good cut.")
                    break

        # Lower error is better
        print(f"Found {len(candidate_cuts)} cuts.")
        candidate_cuts.sort(key=lambda x: x[1])
        return [c[0] for c in candidate_cuts]

    def generate_and_filter_cuts_convex(self, piece: Polygon, children: int) -> list:
        """
        Sweeps from centroid outward for each angle.
        Assumes piece is convex, so each sweep line intersects boundary at exactly two points.
        Also, the area will increase monotonically as we sweep outward from centroid.
        Thus, we can use binary search.
        """
        # Collect all boundary coordinates once
        coords = list(piece.exterior.coords)
        if piece.interiors:
            for interior in piece.interiors:
                coords.extend(list(interior.coords))

        if piece.area < 2 * self.final_piece_min_area:
            print(f"Piece too small to cut, area {piece.area}.")
            return []

        candidate_cuts = []
        cut_angles = []
        degrees_per_cut = 180 / self.N_SWEEP_DIRECTIONS
        for i in range(self.N_SWEEP_DIRECTIONS):
            cut_angles.append(i * degrees_per_cut)
        for angle_deg in cut_angles:
            angle_rad = math.radians(angle_deg)

            # Normal vector (direction we sweep along)
            normal_x = math.cos(angle_rad)
            normal_y = math.sin(angle_rad)

            # Line direction (perpendicular to normal)
            line_dx = -math.sin(angle_rad)
            line_dy = math.cos(angle_rad)

            # Project all boundary points onto the normal axis to find sweep range
            projections = [x * normal_x + y * normal_y for x, y in coords]
            min_proj = min(projections)
            max_proj = max(projections)

            # Binary search for a cut that creates valid areas
            # We want: min_area <= piece1_area <= total_area - min_area
            left = min_proj
            right = max_proj
            tolerance = 0.01  # 0.01 cm tolerance

            search_direction_determined = False
            piece1_grows_with_projection = True  # will be set on first iteration

            while right - left > tolerance:
                mid = (left + right) / 2

                # Create a line at this projection
                center_x = mid * normal_x
                center_y = mid * normal_y

                # Extend the line far enough to cover the polygon
                extension = 1000
                p1 = (center_x - extension * line_dx, center_y - extension * line_dy)
                p2 = (center_x + extension * line_dx, center_y + extension * line_dy)

                sweep_line = LineString([p1, p2])
                intersection = sweep_line.intersection(piece.boundary)

                if intersection.is_empty:
                    # Shouldn't happen for valid convex polygons
                    break

                # Extract points from the intersection
                points: list[Point] = []
                if intersection.geom_type == "Point":
                    points = [intersection]
                elif intersection.geom_type == "MultiPoint":
                    points = list(intersection.geoms)
                elif intersection.geom_type == "GeometryCollection":
                    points = [
                        geom for geom in intersection.geoms if geom.geom_type == "Point"
                    ]

                # For convex polygons, we should get exactly 2 intersection points
                if len(points) != 2:
                    break

                from_p: Point = points[0]
                to_p: Point = points[1]
                line = LineString([from_p, to_p])

                try:
                    split_result = split(piece, line)
                    if len(split_result.geoms) != 2:
                        break

                    piece1, piece2 = split_result.geoms
                    piece1_area = piece1.area
                    piece2_area = piece2.area

                    # On first iteration, determine which piece grows with increasing projection
                    if not search_direction_determined:
                        # Try a slightly higher projection to see which piece grows
                        test_proj = mid + tolerance * 2
                        test_center_x = test_proj * normal_x
                        test_center_y = test_proj * normal_y
                        test_p1 = (
                            test_center_x - extension * line_dx,
                            test_center_y - extension * line_dy,
                        )
                        test_p2 = (
                            test_center_x + extension * line_dx,
                            test_center_y + extension * line_dy,
                        )
                        test_line = LineString([test_p1, test_p2])

                        try:
                            test_split = split(piece, test_line)
                            if len(test_split.geoms) == 2:
                                test_piece1, test_piece2 = test_split.geoms
                                # If piece1 area increased, piece1 grows with projection
                                piece1_grows_with_projection = (
                                    test_piece1.area > piece1_area
                                )
                                search_direction_determined = True
                        except:  # noqa: E722
                            pass

                    # Check if both pieces satisfy minimum area constraint
                    if (
                        piece1_area >= self.final_piece_min_area
                        and piece2_area >= self.final_piece_min_area
                    ):
                        # Found a valid cut! Calculate ratio and store it
                        piece1_ratio = self.cake.get_piece_ratio(piece1)
                        piece2_ratio = self.cake.get_piece_ratio(piece2)

                        ratio1_error = abs(piece1_ratio - self.final_piece_target_ratio)
                        ratio2_error = abs(piece2_ratio - self.final_piece_target_ratio)
                        ratio_error = ratio1_error + ratio2_error

                        candidate_cuts.append(((from_p, to_p), ratio_error))
                        break  # Early terminate for this angle

                    # Binary search logic: adjust search space based on which constraint is violated
                    if piece1_area < self.final_piece_min_area:
                        # piece1 is too small
                        if piece1_grows_with_projection:
                            left = mid  # Move forward to grow piece1
                        else:
                            right = mid  # Move backward to grow piece1
                    else:
                        # piece2 is too small
                        if piece1_grows_with_projection:
                            right = mid  # Move backward to grow piece2
                        else:
                            left = mid  # Move forward to grow piece2

                except Exception:
                    # If we can't split properly, this position doesn't work
                    # Try moving toward a more central position
                    if mid < (min_proj + max_proj) / 2:
                        left = mid
                    else:
                        right = mid

        print(
            f"Found {len(candidate_cuts)} cuts using binary search (convex optimization)."
        )
        # Lower error is better
        candidate_cuts.sort(key=lambda x: x[1])
        return [c[0] for c in candidate_cuts]
