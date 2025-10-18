from shapely import Point, LineString, MultiPoint
from shapely.geometry import Polygon
from shapely.ops import split
from typing import cast
from math import radians, sin, cos, pi
import src.constants as c
from players.player import Player
from players.random_player import RandomPlayer
from src.cake import Cake, extend_line, copy_geom


class CrustOptimizingPlayer(Player):
    """
    Strategy:
    - Samples random points along the cake crust (outer edge)
    - Computes crust density to find crust-heavy regions
    - Picks starting point p1 from densest crust points
    - Chooses p2 and evaluates cut quality using weighted scoring
    - Includes an angle-based fallback for precision
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.random_player = RandomPlayer(children, cake, cake_path)
        self.valid_line_pair = dict()
        self.tolerance = 0.05
        # FIX: Added the missing attribute initialization
        self.target_area_tolerance = 0.0005
        self.max_crust_points = 3
        self.angle_increment = 15  # Degrees
        self.offset = max(min(100 * self.children, 900), 500)
        self.num_candidates = 100

    # ---------- Utility methods ----------

    def _get_pieces_after_temp_cut(
        self, piece: Polygon, from_p: Point, to_p: Point
    ) -> list[Polygon]:
        """
        Simulates a cut on a given piece and returns the resulting pieces.
        This function replaces the non-existent method in the Cake class.
        """
        piece_copy = copy_geom(piece)

        # FIX: Ensure from_p and to_p are Shapely Point objects
        if not isinstance(from_p, Point):
            from_p = Point(from_p)
        if not isinstance(to_p, Point):
            to_p = Point(to_p)

        bound = piece_copy.boundary
        a = bound.interpolate(bound.project(from_p))
        b = bound.interpolate(bound.project(to_p))

        line = LineString([a, b])
        line = extend_line(line)

        try:
            split_piece = split(piece_copy, line)
            split_pieces: list[Polygon] = [
                cast(Polygon, geom) for geom in split_piece.geoms
            ]
            return split_pieces
        except Exception:
            return []

    def get_crust_length(self, piece: Polygon) -> float:
        """Length of piece boundary that lies on the exterior crust."""
        return piece.boundary.intersection(self.cake.exterior_shape.boundary).length

    def get_crust_density(self, piece: Polygon, p: Point):
        """Approximate local crust density near a boundary point."""
        crust_section = self.cake.exterior_shape.boundary.buffer(c.CRUST_SIZE * 2)
        nearby = piece.boundary.intersection(crust_section)
        return nearby.length

    def get_piece(self, p1: Point, p2: Point) -> Polygon | None:
        """Find which piece both points belong to."""
        for piece in self.cake.get_pieces():
            if self.cake.point_lies_on_piece_boundary(
                p1, piece
            ) and self.cake.point_lies_on_piece_boundary(p2, piece):
                return piece
        return None

    def find_cuts(self, line: LineString, piece: Polygon) -> tuple[Point, Point] | None:
        """
        Finds the first two valid intersection points between a line and a piece's boundary.
        Returns the two points or None if a valid two-point cut cannot be formed.
        """
        intersection = line.intersection(piece.boundary)

        if intersection.is_empty:
            return None

        if isinstance(intersection, Point):
            return None

        if isinstance(intersection, LineString):
            return None

        if isinstance(intersection, MultiPoint):
            points = list(intersection.geoms)
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    p1 = points[i]
                    p2 = points[j]

                    is_valid, _ = self.cake.cut_is_valid(p1, p2)

                    if is_valid:
                        return (p1, p2)

        return None

    def check_precision(
        self,
        p1: Point,
        p2: Point,
        Area_list: list[float],
        piece: Polygon,
        crust_ratio: float,
    ) -> tuple[float, float]:
        """Returns precision metrics using the new utility method."""
        split_pieces = self._get_pieces_after_temp_cut(piece, p1, p2)

        if len(split_pieces) < 2:
            return float("inf"), float("inf")

        # Calculate metrics for the first piece
        piece1 = split_pieces[0]
        crust_precision_1 = abs(self.cake.get_piece_ratio(piece1) - crust_ratio)
        closest_target_area_1 = min(Area_list, key=lambda a: abs(a - piece1.area))
        cake_precision_1 = (
            abs(piece1.area - closest_target_area_1) / self.cake.exterior_shape.area
        )

        # Calculate metrics for the second piece
        piece2 = split_pieces[1]
        crust_precision_2 = abs(self.cake.get_piece_ratio(piece2) - crust_ratio)
        closest_target_area_2 = min(Area_list, key=lambda a: abs(a - piece2.area))
        cake_precision_2 = (
            abs(piece2.area - closest_target_area_2) / self.cake.exterior_shape.area
        )

        # Return the maximum of the two for both cake and crust precision
        return max(cake_precision_1, cake_precision_2), max(
            crust_precision_1, crust_precision_2
        )

    def get_weight(
        self, p1: Point, p2: Point, piece: Polygon, Goal_ratio: float
    ) -> float:
        """Compute weight for a given cut — based on area ratio deviation."""
        split_pieces = self._get_pieces_after_temp_cut(piece, p1, p2)
        if len(split_pieces) != 2:
            return 0.0
        r1 = self.cake.get_piece_ratio(split_pieces[0])
        r2 = self.cake.get_piece_ratio(split_pieces[1])
        weight = 0.5 * (1 - abs(Goal_ratio - r1)) + 0.5 * (1 - abs(Goal_ratio - r2))
        return weight

    def _get_line_at_angle(
        self, center_x: float, center_y: float, angle_degrees: float, length: float
    ) -> LineString:
        """Creates a line of a given length, angle, and center point."""
        angle_rad = radians(angle_degrees)
        half_len = length / 2
        p1_x = center_x - half_len * cos(angle_rad)
        p1_y = center_y - half_len * sin(angle_rad)
        p2_x = center_x + half_len * cos(angle_rad)
        p2_y = center_y + half_len * sin(angle_rad)
        return LineString([(p1_x, p1_y), (p2_x, p2_y)])

    def _calculate_area_at_offset(
        self, piece: Polygon, offset: float, angle_degrees: float
    ) -> float:
        """Calculates the area of one side of a cut at a specific offset and angle."""
        bounds = piece.bounds
        search_range = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2

        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

        angle_rad = radians(angle_degrees + 90)
        cut_center_x = center_x + offset * cos(angle_rad)
        cut_center_y = center_y + offset * sin(angle_rad)

        line = self._get_line_at_angle(
            cut_center_x, cut_center_y, angle_degrees, search_range
        )

        split_pieces = self._get_pieces_after_temp_cut(
            piece, line.coords[0], line.coords[1]
        )
        if len(split_pieces) < 2:
            return 0.0
        return min(p.area for p in split_pieces)

    def _find_all_cuts_by_angle(
        self, piece: Polygon, Area_list: list[float], crust_ratio: float
    ) -> list[tuple]:
        """Iterates through angles and tests every cut to find those that meet precision requirements."""
        all_found_cuts = []

        # Iterate through angles
        for angle in range(0, 180, self.angle_increment):
            bounds = piece.bounds
            search_range = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2

            # Brute-force offsets
            num_offsets = self.offset
            step_size = search_range / num_offsets

            for i in range(num_offsets):
                offset = -search_range / 2 + i * step_size

                cut_points = self._get_cut_points_from_offset(piece, offset, angle)

                if cut_points:
                    from_p, to_p = cut_points
                    valid, _ = self.cake.cut_is_valid(from_p, to_p)
                    if valid:
                        cake_p, crust_p = self.check_precision(
                            from_p, to_p, Area_list, piece, crust_ratio
                        )

                        # Test against all possible target areas to see if the precision is good enough
                        if cake_p < self.target_area_tolerance:
                            all_found_cuts.append((cake_p, from_p, to_p, crust_p))

        return all_found_cuts

    def _get_cut_points_from_offset(self, piece, offset, angle):
        bounds = piece.bounds
        search_range = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        angle_rad = radians(angle + 90)
        cut_center_x = center_x + offset * cos(angle_rad)
        cut_center_y = center_y + offset * sin(angle_rad)
        line = self._get_line_at_angle(cut_center_x, cut_center_y, angle, search_range)
        return self.find_cuts(line, piece)

    # ---------- Binary Search Algorithm ----------

    def point_side(self, p1, p2, p3):
        # Computes if p3 is on the left or right of line p1-p2
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y

        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

        if cross > 0:
            return "left"
        elif cross < 0:
            return "right"
        else:
            return "collinear"

    def line_bound_given_PA(self, mid, piece, degree):
        left_bound, below_bound, right_bound, above_bound = piece.bounds
        # print("l, b, r, a ", left_bound, below_bound, right_bound, above_bound)
        upper_left = (left_bound, above_bound)
        width = right_bound - left_bound
        height = above_bound - below_bound
        bar_dimension = (width**2 + height**2) ** 0.5

        # First, we find the boundary of the cake, then starting from
        # the upper left, we find the distance to the lower right,
        # this is the bar_dimension varaible. At this point, we try to cut
        # at mid*bar_dimension starting from the upper right point
        # with the correct degree.

        rad = radians(degree)
        rad_perp = rad + pi / 2

        # Initial point for our line
        point_x = upper_left[0] + mid * bar_dimension * cos(rad_perp)
        point_y = upper_left[1] - mid * bar_dimension * sin(rad_perp)

        # These give the endpoints, Note that since we multiply by bar_dimension, this will be a huge line
        offx = cos(rad) * bar_dimension
        offy = sin(rad) * bar_dimension

        p1 = (point_x - offx, point_y - offy)
        p2 = (point_x + offx, point_y + offy)
        cut_line = LineString([p1, p2])
        return cut_line, p1, p2

    def find_actual_cuts(self, line, piece):
        intersection = line.intersection(piece.boundary)

        if intersection.is_empty:
            return False, 0, 0
        elif intersection.geom_type == "Point":
            return False, 0, 0

        points = []
        if intersection.geom_type == "LineString":
            return False, 0, 0
        elif intersection.geom_type == "MultiPoint":
            points = list(intersection.geoms)

        if len(points) < 2:
            return False, 0, 0

        return True, points[0], points[1]

    def get_area_with_scaler(self, mid, piece, degree):
        line, p1, p2 = self.line_bound_given_PA(mid, piece, degree)
        # This first one find a "big sweeping cut" line that just cuts at the right degree and mid percentage

        found, p1, p2 = self.find_actual_cuts(line, piece)
        # Given that first line, we find the actual endpoints

        if not found:
            return False, 0, 0, 0

        pieces = split(piece, line)

        if len(pieces.geoms) != 2:
            return False, 0, 0, 0

        piece1, piece2 = pieces.geoms

        centroid1 = piece1.centroid

        if self.point_side(Point(p1), Point(p2), centroid1) == "left":
            return True, piece1.area, Point(p1), Point(p2)
        else:
            return True, piece2.area, Point(p1), Point(p2)

    def Search_for_angle_area(self, degree, targt_area, piece, Area_list, crust_ratio):
        # Search for a cut with with valid area and cut made at set degree

        lower_scaler = 0
        upper_scaler = 1

        for _ in range(15):
            # Mid is the "percent" if the way through we cut the cake that we set our angle
            mid = (lower_scaler + upper_scaler) / 2
            valid_BS, area, p1, p2 = self.get_area_with_scaler(mid, piece, degree)

            if not valid_BS:
                lower_scaler = mid
            if area - targt_area < 0.0001:
                break
            if area < targt_area:
                lower_scaler = mid
            elif area > targt_area:
                upper_scaler = mid
            else:
                break

        # Serch for a piece with the set midpoint. If we find one, we run the same code from the standard algorithm
        valid_BS, area, p1, p2 = self.get_area_with_scaler(mid, piece, degree)
        if not valid_BS:
            return False, 0

        valid, _ = self.cake.cut_is_valid(p1, p2)
        if not valid:
            # print(p1, p2)
            return False, 0
        # print(p1,p2)

        cake_precision, crust_precision = self.check_precision(
            p1, p2, Area_list, piece, crust_ratio
        )
        if cake_precision == float("inf"):
            return False, 0

        if cake_precision < self.target_area_tolerance:
            # print(cake_precision)
            return True, (cake_precision, p1, p2, crust_precision)
        return False, 0

    def Binary_Search(self, Area_list, piece, crust_ratio):
        degree_list = []
        step = 0.1
        for i in range(int((360 - 271) / step)):
            degree_list += [271 + step * i]

        valid_cuts = []
        for degree in degree_list:
            for area in Area_list:
                # Search for a cut with with valid area and cut made at set degree
                found, cut = self.Search_for_angle_area(
                    degree, area, piece, Area_list, crust_ratio
                )
                if found and cut not in valid_cuts:
                    valid_cuts += [cut]
        return valid_cuts

    # ---------- Main Algorithm ----------

    def get_cuts(self) -> list[tuple[Point, Point]]:
        moves: list[tuple[Point, Point]] = []

        piece = max(self.cake.get_pieces(), key=lambda p: p.area)
        Total_Area = self.cake.exterior_shape.area
        Area_list = [(Total_Area / self.children) * i for i in range(1, self.children)]

        for k in range(self.children - 1):
            print("-------------------")
            print(f"Cut {k + 1}/{self.children - 1}")
            best_line_list = []
            best_line = [100, None, None, 100]
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            crust_ratio = self.cake.get_piece_ratio(piece)
            piece_boundary = piece.boundary

            num_candidates = self.num_candidates
            step_size = piece_boundary.length / num_candidates
            candidates = [
                piece_boundary.interpolate(i * step_size) for i in range(num_candidates)
            ]

            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    p1 = candidates[i]
                    p2 = candidates[j]

                    good, _ = self.cake.does_line_cut_piece_well(
                        LineString((p1, p2)), piece
                    )
                    if not good:
                        continue

                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        continue

                    cake_precision, crust_precision = self.check_precision(
                        p1, p2, Area_list, piece, crust_ratio
                    )
                    if cake_precision == float("inf"):
                        continue

                    if best_line[0] > cake_precision:
                        best_line = [cake_precision, p1, p2, crust_precision]

                    if cake_precision < self.target_area_tolerance:
                        best_line_list.append((cake_precision, p1, p2, crust_precision))

            print(f"Found {len(best_line_list)} candidates from the first algorithm.")

            if best_line_list:
                # Check the best cut's cake precision from the primary search
                best_primary_cut = min(best_line_list, key=lambda x: x[3])
                f"Primary search found cuts, the best precision ({best_primary_cut[3]:.6f}) is not optimal. Running angle-based fallback."

            all_found_cuts = self._find_all_cuts_by_angle(piece, Area_list, crust_ratio)
            if all_found_cuts:
                best_line_list.extend(all_found_cuts)
                print(f"angle bash search found {len(all_found_cuts)} suitable cuts.")
            else:
                print("Angle-based search failed to find a suitable cut.")

            all_angle_bst_found_cuts = self.Binary_Search(Area_list, piece, crust_ratio)
            if all_angle_bst_found_cuts:
                best_line_list.extend(all_angle_bst_found_cuts)
                # sorted_data = sorted(all_angle_bst_found_cuts, key=lambda x: x[0], reverse=True)
                print(
                    f"Angle-based binary search found {len(all_angle_bst_found_cuts)} suitable cuts."
                )
            else:
                print("Angle-based binary search failed to find a suitable cut.")

            if best_line_list:
                best_line_list.sort(key=lambda x: x[3])
                bestline = best_line_list[0]
                print(
                    "Using best line from candidates or angle search or angle binary search."
                )
                print(
                    f"Cake Precision: {bestline[0]:.6f}, Crust Precision: {bestline[3]:.6f}"
                )
            elif best_line[1] is not None:
                print("No optimal crust found, using best-effort cut.")
                bestline = best_line
            else:
                print("Random strategy as a last resort.")
                a, b = self.random_player.find_random_cut()
                bestline = [100, a, b, 100]

            print(self.num_candidates, self.offset)
            self.num_candidates = self.num_candidates + (450 // self.children)
            self.offset = max(
                int(self.offset * ((self.children - 2) / self.children)), 100
            )
            self.offset = max(
                int(self.offset * ((self.children - 2) / self.children)), 100
            )

            moves.append((bestline[1], bestline[2]))
            self.cake.cut(bestline[1], bestline[2])

        return moves
