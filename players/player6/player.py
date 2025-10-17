from dataclasses import dataclass
from players.player import Player
from src.cake import Cake
from shapely.geometry import Polygon, LineString, Point
from shapely import MultiLineString, intersection
from shapely.ops import split
from typing import cast, List, Optional
from math import hypot, pi, cos, sin, isclose, floor, ceil
import numpy as np
from joblib import Parallel, delayed
import src.constants as c


@dataclass
class CutResult:
    polygons: list[Polygon]
    points: tuple[Point, Point]


def extend_line(line: LineString, fraction: float = 0.05) -> LineString:
    coords = list(line.coords)
    if len(coords) != 2:
        return line

    (x1, y1), (x2, y2) = coords
    dx, dy = (x2 - x1), (y2 - y1)
    L = hypot(dx, dy)
    if L == 0:
        return line

    ux, uy = dx / L, dy / L

    a = fraction * L

    x1n, y1n = x1 - a * ux, y1 - a * uy
    x2n, y2n = x2 + a * ux, y2 + a * uy

    return LineString([(x1n, y1n), (x2n, y2n)])


class Player6(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.target_area = self.cake.get_area() / self.children
        self.target_ratio = (
            self.cake.interior_shape.area / self.cake.exterior_shape.area
        )

    def find_common_points(self, pieces: list[Polygon]) -> tuple[tuple[float, float]]:
        """Find the vertices of the new cake polygon once cut"""
        return tuple(set(pieces[0].exterior.coords) & set(pieces[1].exterior.coords))

    def __cut_is_within_cake(self, cut: LineString, polygon: Polygon) -> bool:
        """Checks cut is within cake"""
        outside = cut.difference(polygon.buffer(c.TOL * 2))
        return outside.is_empty

    def get_intersecting_pieces_from_point(self, p: Point, polygons: list[Polygon]):
        """Returns list of polygons whose boundary touches the specified point"""
        touched_pieces = [
            piece for piece in polygons if self.point_lies_on_piece_boundary(p, piece)
        ]

        return touched_pieces

    def point_lies_on_piece_boundary(self, p: Point, piece: Polygon):
        """Checks whether a point is within tolerance of a piece's boundary"""
        return p.distance(piece.boundary) <= c.TOL

    def get_cuttable_piece(self, from_p: Point, to_p: Point, polygons: list[Polygon]):
        """Determines if a polygon can be cut with the specified line"""
        a_pieces = self.get_intersecting_pieces_from_point(from_p, polygons)
        b_pieces = self.get_intersecting_pieces_from_point(to_p, polygons)

        contenders = set(a_pieces).intersection(set(b_pieces))

        if len(contenders) > 1:
            return None, "line can cut multiple pieces, should only cut one"

        if len(contenders) == 0:
            return None, "line doesn't cut any piece of cake well"

        piece = list(contenders)[0]

        # snap points to piece boundary
        bound = piece.boundary
        a = bound.interpolate(bound.project(from_p))
        b = bound.interpolate(bound.project(to_p))

        line = extend_line(LineString([a, b]))

        piece_well_cut, reason = self.does_line_cut_piece_well(line, piece)
        if not piece_well_cut:
            return None, reason

        return piece, ""

    def does_line_cut_piece_well(self, line: LineString, piece: Polygon):
        """Checks whether line cuts piece in two valid (large enough) pieces"""
        if piece.touches(line):
            return False, "cut lies on piece boundary"

        if not line.crosses(piece):
            return False, "line does not cut through piece"

        cut_pieces = split(piece, line)
        if len(cut_pieces.geoms) != 2:
            return False, f"line cuts piece in {len(cut_pieces.geoms)}, not 2"

        all_sizes_are_good = all([p.area >= c.MIN_PIECE_AREA for p in cut_pieces.geoms])

        if not all_sizes_are_good:
            return False, "line cuts a piece that's too small"

        return True, ""

    def cut_is_valid(
        self, from_p: Point, to_p: Point, polygon: Polygon
    ) -> tuple[bool, str]:
        """Check whether a cut from `from_p` to `to_p` is valid.

        If invalid, the method returns the reason as the second argument.
        """
        line = LineString([from_p, to_p])

        if not self.__cut_is_within_cake(line, polygon):
            return False, "cut is not within cake"

        cuttable_piece, reason = self.get_cuttable_piece(from_p, to_p, [polygon])

        if not cuttable_piece:
            return False, reason

        return True, "valid"

    def virtual_cut(self, piece: Polygon, cut: LineString) -> CutResult | None:
        """
        Make a virtual cut over our existing cake(piece)
        """

        # --------
        # >>> from shapely import LineString
        # >>> x, y = LineString([(0, 0), (1, 1)]).xy
        # >>> list(x)
        # [0.0, 1.0]
        # >>> list(y)
        # [0.0, 1.0]
        # x, y = cut.coords
        coords = list(cut.coords)
        from_p, to_p = Point(coords[0]), Point(coords[1])

        is_valid, reason = self.cut_is_valid(from_p, to_p, piece)
        # NOTE: catch this later
        if not is_valid:
            return None

        # as this cut is valid, we will have exactly one cuttable piece
        target_piece, _ = self.get_cuttable_piece(from_p, to_p, [piece])
        if not target_piece:
            return None

        bound = piece.boundary
        a = bound.interpolate(bound.project(from_p))
        b = bound.interpolate(bound.project(to_p))

        line = LineString([a, b])
        # ensure that the line extends beyond the piece
        line = extend_line(line)
        # cut the cake
        split_piece = split(piece, line)
        split_pieces: list[Polygon] = [
            cast(Polygon, geom) for geom in split_piece.geoms
        ]
        # point1, point2 = line.coords
        # output = CutResult(polygons=split_pieces, points=(Point(point1), Point(point2)))
        output = CutResult(polygons=split_pieces, points=(a, b))
        return output

    def current_polygon(self, cut: CutResult) -> set[Polygon]:
        """Returns the two polygons involved in the cut based on their shared points"""
        cut_points, polygons = cut.points, cut.polygons
        res: set[Polygon] = set()
        for polygon in polygons:
            for x, y in polygon.exterior.coords:
                for point in cut_points:
                    if point.x == x and point.y == y:
                        res.add(polygon)
        return res

    def get_piece_ratio(self, piece: Polygon):
        """Calculate ratio of the piece's area that overlaps with the interior cake shape"""
        if piece.intersects(self.cake.interior_shape):
            inter = piece.intersection(self.cake.interior_shape)
            return inter.area / piece.area if not inter.is_empty else 0
        return 0

    def score_cut(self, cut: CutResult) -> tuple[float, float]:
        """Calculate the score of a cut with stddev from target area and cake to crust ratio"""
        if cut is None:
            return (float("inf"), float("inf"))

        polygons = self.current_polygon(cut)

        if len(polygons) != 2:
            return (float("inf"), float("inf"))

        area_scores = [abs(polygon.area - self.target_area) for polygon in polygons]
        ratio_scores = [
            abs(self.get_piece_ratio(polygon) - self.target_ratio)
            for polygon in polygons
        ]

        area_score = min(area_scores)
        ratio_score = ratio_scores[area_scores.index(min(area_scores))]

        # If difference from target area < 0.125, treat it as equal → rely on ratio
        if area_score <= 0.125:
            area_score = 0.0
        if ratio_score <= 0.05:
            ratio_score = 0.0

        return (area_score, ratio_score)

    def positions_best_cut(
        self,
        try_fn,
        position: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        piece: Polygon,
    ) -> tuple[CutResult, tuple[float, float]]:
        """Calculates the best cut and score based on all possible cuts at a given position"""
        cuts = try_fn(position, min_x, max_x, min_y, max_y, piece)
        best_cut = None
        best_score = (float("inf"), float("inf"))
        if cuts:
            for cut in cuts:
                if cut is not None:
                    score = self.score_cut(cut)
                    # NOTE: these are tuples we need to check both else area dominates
                    if score < best_score:
                        best_score, best_cut = score, cut

        # NOTE: catch later if invalid
        return best_cut, best_score

    def ternary_search_cut(
        self,
        try_fn,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        piece: Polygon,
        epsilon: float = 0.000001,
    ) -> tuple[CutResult, tuple[float, float]]:
        """Ternary search for the cut positio based on the slicing function to try, returns best cut and its score"""
        left, right = 0.01, 1
        best_cut, best_score = None, (float("inf"), float("inf"))
        iterations = 0
        # NOTE: tune later
        max_iterations = 2000

        while right - left > epsilon and iterations < max_iterations:
            iterations += 1

            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3

            cut1, score1 = self.positions_best_cut(
                try_fn, mid1, min_x, max_x, min_y, max_y, piece
            )
            cut2, score2 = self.positions_best_cut(
                try_fn, mid2, min_x, max_x, min_y, max_y, piece
            )

            if score1 < best_score:
                best_cut, best_score = cut1, score1
            if score2 < best_score:
                best_cut, best_score = cut2, score2

            if score1 < score2:
                right = mid2
            else:
                left = mid1

        return best_cut, best_score

    def intersect_cut_line(
        self, piece: Polygon, line: LineString
    ) -> Optional[List["CutResult"]]:
        """
        Intersect the line with the polygon and return virtual cuts.
        Returns None if no intersection.
        """
        intersections = intersection(line, piece)
        if intersections.is_empty:
            return None

        results: List[CutResult] = []

        if isinstance(intersections, MultiLineString):
            # line cuts at multiple positions
            for seg in intersections.geoms:
                cut_res = self.virtual_cut(piece, seg)
                if cut_res:
                    results.append(cut_res)

        if isinstance(intersections, LineString):
            # line cuts at one position
            cut_res = self.virtual_cut(piece, intersections)
            if cut_res:
                results.append(cut_res)

        return results

    def generate_cut_line(
        self, piece: Polygon, angle: float, frac: float
    ) -> LineString:
        """
        Generate a LineString for a given angle (fraction of pi) and sweep fraction.
        Handles vertical and horizontal lines robustly.
        """
        min_x, min_y, max_x, max_y = piece.bounds
        theta = angle * pi

        # When cos(theta) is very small → vertical line
        if isclose(abs(angle), 0.5, abs_tol=0.01):
            # Use x = constant line
            # print("ANGLE IS CLOSE TO 90 ")
            # print(min_x, min_y, max_x, max_y)
            x_const = min_x + frac * (max_x - min_x)
            return LineString([(x_const, min_y), (x_const, max_y)])
        else:
            # For angled lines, we need to create a line that sweeps across the bounding box
            # Use parametric approach: start from one edge and go to the opposite edge

            # Calculate the center point and use it as reference
            center_x = min_x + frac * (max_x - min_x)
            center_y = min_y + frac * (max_y - min_y)

            # Create a line through the center point with the given angle
            # Line direction vector from angle
            dx = cos(theta)
            dy = sin(theta)

            # Extend the line to intersect the bounding box
            # Calculate how far we need to extend to reach the edges
            width = max_x - min_x
            height = max_y - min_y
            max_distance = hypot(width, height)

            # Create line endpoints extending in both directions
            x1 = center_x - max_distance * dx
            y1 = center_y - max_distance * dy
            x2 = center_x + max_distance * dx
            y2 = center_y + max_distance * dy

            return LineString([(x1, y1), (x2, y2)])

    def _try_angle_slice(
        self,
        position: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        piece: Polygon,
        angle: float = 0.0,
    ) -> list[CutResult] | None:
        """Attempt to make a cut at the given angle and position."""
        line = self.generate_cut_line(piece, angle, position)
        return self.intersect_cut_line(piece, line)

    def _evaluate_angle(
        self,
        angle: float,
        largest_piece: Polygon,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> tuple[CutResult, tuple[float, float]]:
        """Evaluate a single angle using ternary search - helper for parallel processing"""
        cut, score = self.ternary_search_cut(
            lambda pos, min_x, max_x, min_y, max_y, piece: self._try_angle_slice(
                pos, min_x, max_x, min_y, max_y, piece, angle
            ),
            min_x,
            max_x,
            min_y,
            max_y,
            largest_piece,
        )
        return cut, score

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Adaptive coarse-to-fine search for cuts with multiple angles using parallel processing."""
        # get_cuts_iterative() for n-1 children
        result = self.get_cuts_divide_conquer()
        if not result:
            return []
        return result

    def get_cuts_iterative(self) -> list[tuple[Point, Point]]:
        result: list[tuple[Point, Point]] = []

        while len(result) < self.children - 1:
            largest_piece = self.get_max_piece(self.cake.exterior_pieces)
            min_x, min_y, max_x, max_y = largest_piece.exterior.bounds

            # Try multiple angles: horizontal (0), vertical (0.5), and some diagonal angles
            angles_to_try = np.linspace(-1, 1, 360)

            # Use parallel processing to evaluate all angles
            results = Parallel(n_jobs=-1)(
                delayed(self._evaluate_angle)(
                    angle, largest_piece, min_x, max_x, min_y, max_y
                )
                for angle in angles_to_try
            )

            # Find the best result from all parallel evaluations
            best_slice, best_score = None, (float("inf"), float("inf"))
            for cut, score in results:
                if cut is not None and score[0] < best_score[0]:
                    best_score, best_slice = score, cut

            if not best_slice:
                break

            if best_slice:
                result.append(best_slice.points)
                self.cake.cut(best_slice.points[0], best_slice.points[1])

        return result

    def get_max_piece(self, pieces: list[Polygon]) -> Polygon:
        """Return largest polygon from list"""
        return max(pieces, key=lambda piece: piece.area)

    def get_cuts_divide_conquer(self) -> list[tuple[Point, Point]]:
        return self.divide_and_conquer(
            self.get_max_piece(self.cake.exterior_pieces), self.children
        )

    def divide_and_conquer(
        self, piece: Polygon, n_children: int
    ) -> list[tuple[Point, Point]]:
        if n_children <= 1:
            # no more cuts to make - either no children or the child gets this piece
            # not none, because we are doing a concatenation - should always be some sort of set
            result: list[tuple[Point, Point]] = []
            return result
        else:
            result: list[tuple[Point, Point]] = []
            # divide the cake into two pieces
            min_x, min_y, max_x, max_y = piece.exterior.bounds
            # Try multiple angles: horizontal (0), vertical (0.5), and some diagonal angles
            angles_to_try = np.linspace(-1, 1, 360)
            # Use parallel processing to evaluate all angles for the cut
            results: List[tuple[CutResult, tuple[float, float]]] = Parallel(n_jobs=-1)(
                delayed(self._evaluate_angle_n)(
                    angle, piece, min_x, max_x, min_y, max_y, n_children
                )
                for angle in angles_to_try
            )
            # Find the best result from all parallel evaluations
            best_slice, best_score = None, (float("inf"), float("inf"))
            for cut, score in results:
                if cut is not None and score < best_score and len(cut.polygons) == 2:
                    best_score, best_slice = score, cut

            if not best_slice:
                # probably another check better
                return result

            cut_res = self.virtual_cut(piece, LineString(best_slice.points))

            if not cut_res:
                return result

            pieces = cut_res.polygons
            pieces.sort(key=lambda piece: piece.area)

            result.append(best_slice.points)
            self.cake.cut(best_slice.points[0], best_slice.points[1])
            large_result = self.divide_and_conquer(pieces[1], ceil(n_children / 2))
            small_result = self.divide_and_conquer(pieces[0], floor(n_children / 2))
            return result + small_result + large_result

    def _evaluate_angle_n(
        self,
        angle: float,
        largest_piece: Polygon,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        n_children: int,
    ) -> tuple[CutResult, tuple[float, float]]:
        """Evaluate a single angle using ternary search - helper for parallel processing"""
        cut, score = self.ternary_search_cut_n(
            lambda pos, min_x, max_x, min_y, max_y, piece: self._try_angle_slice(
                pos, min_x, max_x, min_y, max_y, piece, angle
            ),
            min_x,
            max_x,
            min_y,
            max_y,
            largest_piece,
            n_children,
        )
        return cut, score

    def ternary_search_cut_n(
        self,
        try_fn,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        piece: Polygon,
        n_children: int,
        epsilon: float = 0.00001,
    ) -> tuple[CutResult, tuple[float, float]]:
        """Ternary search for the cut positio based on the slicing function to try, returns best cut and its score"""
        left, right = 0.01, 0.99
        best_cut, best_score = None, (float("inf"), float("inf"))
        iterations = 0
        # NOTE: tune later
        max_iterations = 2000

        while right - left > epsilon and iterations < max_iterations:
            iterations += 1

            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3

            cut1, score1 = self.positions_best_cut_n(
                try_fn, mid1, min_x, max_x, min_y, max_y, piece, n_children
            )
            cut2, score2 = self.positions_best_cut_n(
                try_fn, mid2, min_x, max_x, min_y, max_y, piece, n_children
            )

            if score1 < best_score:
                best_cut, best_score = cut1, score1
            if score2 < best_score:
                best_cut, best_score = cut2, score2

            if score1 < score2:
                right = mid2
            else:
                left = mid1

        return best_cut, best_score

    def positions_best_cut_n(
        self,
        try_fn,
        position: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        piece: Polygon,
        n_children: int,
    ) -> tuple[CutResult, tuple[float, float]]:
        """Calculates the best cut and score based on all possible cuts at a given position"""
        cuts = try_fn(position, min_x, max_x, min_y, max_y, piece)
        best_cut = None
        best_score = (float("inf"), float("inf"))
        if cuts:
            for cut in cuts:
                if cut is not None:
                    score = self.score_cut_n(cut, n_children)
                    # NOTE: these are tuples we need to check both else area dominates
                    if score < best_score:
                        best_score, best_cut = score, cut

        # NOTE: catch later if invalid
        return best_cut, best_score

    def score_cut_n(self, cut: CutResult, n_children: int) -> tuple[float, float]:
        """Calculate the score of a cut with stddev from target area and cake to crust ratio"""
        if cut is None:
            return (float("inf"), float("inf"))

        polygons = self.current_polygon(cut)
        # polygons.sort(key = lambda piece: piece.area)
        if len(polygons) != 2:
            return (float("inf"), float("inf"))
        areas = [polygon.area for polygon in polygons]
        small_area_score = abs(min(areas) / floor(n_children / 2) - self.target_area)
        large_area_score = abs(max(areas) / ceil(n_children / 2) - self.target_area)
        # area_scores = [abs(polygon.area - self.target_area) for polygon in polygons]
        area_scores = [small_area_score, large_area_score]
        ratio_scores = [
            abs(self.get_piece_ratio(polygon) - self.target_ratio) / self.target_ratio
            for polygon in polygons
        ]
        # always looking at largest area error - never want it to be too big
        area_score = max(area_scores)
        # always looking at largest ratio - since we check everything for ternary search?
        ratio_score = max(ratio_scores)
        # ratio_score = ratio_scores[area_scores.index(min(area_scores))]

        # If difference from target area < 0.125, treat it as equal → rely on ratio
        if area_score < 0.245:
            area_score = 0.0
        if ratio_score <= 0.015:
            ratio_score = 0.0

        # adding line length as the last factor as after dividing area equally we would love to have more interior !
        return (area_score, ratio_score, 1 / LineString(cut.points).length)
