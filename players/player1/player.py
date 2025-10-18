from dataclasses import dataclass
from players.player import Player
from src.cake import Cake
from shapely.geometry import Polygon, LineString, Point
from shapely import MultiLineString, intersection
from shapely.ops import split
from typing import cast, List
from math import (
    hypot,
    pi,
    cos,
    sin,
    isclose,
    floor,
    ceil,
    sqrt,
    isclose as math_isclose,
)
import numpy as np
from joblib import Parallel, delayed
import src.constants as c

# Golden ratio constant for search
PHI = (1 + sqrt(5)) / 2
RES_TUPLE = tuple[float, float, float]  # (Area_Score, Ratio_Score, Length_Score)


@dataclass
class CutResult:
    """Stores the resulting polygons and the cut endpoints."""

    polygons: list[Polygon]
    points: tuple[Point, Point]


def extend_line_robust(line: LineString, extension_factor: float = 1.1) -> LineString:
    """
    Extends a LineString from its center point outward by an extension_factor
    relative to its original length, ensuring it spans beyond the piece.
    """
    coords = list(line.coords)
    if len(coords) != 2:
        return line

    (x1, y1), (x2, y2) = coords
    dx, dy = (x2 - x1), (y2 - y1)
    L = hypot(dx, dy)
    if L == 0:
        return line

    # Unit vector
    ux, uy = dx / L, dy / L

    # New length
    new_L = L * extension_factor

    # Center point
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # New endpoints extending from the center
    x1n, y1n = cx - new_L / 2 * ux, cy - new_L / 2 * uy
    x2n, y2n = cx + new_L / 2 * ux, cy + new_L / 2 * uy

    return LineString([(x1n, y1n), (x2n, y2n)])


class Player1(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.target_area = self.cake.get_area() / self.children
        # ratio target isn't directly used in the metric selection now, but kept for scoring ties
        self.target_ratio = self.cake.get_piece_ratio(self.cake.exterior_shape)

        # “threshold” you asked for — used as a flat valley on area error in cut scoring
        self.flat_tol: float = 0.0

    # --- Utility Methods ---

    def _get_piece_boundary_touchers(
        self, p: Point, pieces: list[Polygon]
    ) -> list[Polygon]:
        """Returns list of polygons whose boundary is near the specified point."""
        return [piece for piece in pieces if p.distance(piece.boundary) <= c.TOL]

    def _is_cut_fully_contained(self, cut: LineString, polygon: Polygon) -> bool:
        """Checks if the cut is entirely within the polygon boundary plus tolerance."""
        return cut.difference(polygon.buffer(c.TOL * 2)).is_empty

    def _get_single_cuttable_piece(self, p1: Point, p2: Point, polygons: list[Polygon]):
        """
        Identifies the single polygon that is meant to be cut.
        Returns: tuple[Polygon, LineString, str] on success, or tuple[None, None, str] on failure.
        """
        pieces_at_p1 = self._get_piece_boundary_touchers(p1, polygons)
        pieces_at_p2 = self._get_piece_boundary_touchers(p2, polygons)

        contenders = set(pieces_at_p1).intersection(set(pieces_at_p2))

        if len(contenders) != 1:
            reason = f"Cut touches {len(contenders)} pieces, must be exactly 1."
            return None, None, reason

        piece = list(contenders)[0]

        # Snap points to piece boundary for the actual cut line
        bound = piece.boundary
        a = bound.interpolate(bound.project(p1))
        b = bound.interpolate(bound.project(p2))

        # Line extended from snapped points
        test_line = extend_line_robust(LineString([a, b]))

        if piece.touches(test_line):
            return None, None, "Extended cut lies on piece boundary."

        if not test_line.crosses(piece):
            return None, None, "Extended line does not cut through piece."

        cut_pieces = split(piece, test_line)
        if len(cut_pieces.geoms) != 2:
            reason = f"Cut results in {len(cut_pieces.geoms)} pieces, not 2."
            return None, None, reason

        # Check for minimum area constraint
        if not all(p.area >= c.MIN_PIECE_AREA for p in cut_pieces.geoms):
            return None, None, "Cut results in a piece that's too small."

        # Success case
        return piece, LineString([a, b]), ""

    def _validate_and_make_cut(
        self, from_p: Point, to_p: Point, piece: Polygon
    ) -> CutResult | None:
        """Validates a cut and executes it virtually if valid."""
        line_segment = LineString([from_p, to_p])
        if not self._is_cut_fully_contained(line_segment, piece):
            return None

        target_piece, snapped_line, reason = self._get_single_cuttable_piece(
            from_p, to_p, [piece]
        )
        if target_piece is None:
            return None

        # Re-extend the snapped line for the split operation
        extended_snapped_line = extend_line_robust(snapped_line)
        split_piece = split(target_piece, extended_snapped_line)
        split_pieces: list[Polygon] = [
            cast(Polygon, geom) for geom in split_piece.geoms
        ]

        # Use the snapped points for the final output
        a, b = Point(snapped_line.coords[0]), Point(snapped_line.coords[1])
        return CutResult(polygons=split_pieces, points=(a, b))

    def _get_piece_ratio(self, piece: Polygon) -> float:
        """Use engine API so ratio matches the simulator/game.py."""
        try:
            return float(self.cake.get_piece_ratio(piece))
        except Exception:
            # very rare fallback
            if piece.intersects(self.cake.interior_shape):
                inter = piece.intersection(self.cake.interior_shape)
                return (
                    inter.area / piece.area
                    if not inter.is_empty and piece.area > 0
                    else 0.0
                )
            return 0.0

    def _score_cut_by_division(
        self, cut: CutResult, n_children_to_divide: int
    ) -> RES_TUPLE:
        """
        Score a candidate cut for a given piece split.
        Primary = area closeness to each side's slice budget (with flat_tol),
        Secondary = ratio proximity,
        Tertiary = prefer shorter chord.
        """
        if cut is None or len(cut.polygons) != 2:
            return (float("inf"), float("inf"), float("inf"))

        p1, p2 = cut.polygons
        areas = [p1.area, p2.area]
        ratios = [self._get_piece_ratio(p1), self._get_piece_ratio(p2)]

        # small/large
        small_idx, large_idx = (0, 1) if areas[0] <= areas[1] else (1, 0)
        small_area, large_area = areas[small_idx], areas[large_idx]

        # expected children per side
        n_small = floor(n_children_to_divide / 2)
        n_large = ceil(n_children_to_divide / 2)

        # target *total* area for each side (not per-child)
        target_small = n_small * self.target_area
        target_large = n_large * self.target_area

        # area error per side
        area_err_small = abs(small_area - target_small) if n_small > 0 else 0.0
        area_err_large = abs(large_area - target_large) if n_large > 0 else 0.0
        area_score = max(area_err_small, area_err_large)

        # apply your threshold (=flat valley): anything within tol is "equally fine"
        if self.flat_tol > 0.0 and area_score < self.flat_tol:
            area_score = 0.0

        # ratio closeness (to global target ratio)
        ratio_score = max(abs(r - self.target_ratio) for r in ratios)

        # prefer shorter chord when ties
        chord_len = LineString(cut.points).length
        length_score = chord_len if chord_len > 0 else float("inf")

        return (area_score, ratio_score, length_score)

    # --- Search machinery ---

    def _generate_line_from_angle_pos(
        self, piece: Polygon, angle_rad: float, position_frac: float
    ) -> LineString:
        """Long line across the piece's bbox at angle angle_rad and sweep position."""
        min_x, min_y, max_x, max_y = piece.bounds

        # Near-vertical
        if isclose(cos(angle_rad), 0.0, abs_tol=0.01):
            x_const = min_x + position_frac * (max_x - min_x)
            return LineString([(x_const, min_y), (x_const, max_y)])

        width = max_x - min_x
        height = max_y - min_y

        ref_x = min_x + position_frac * width
        ref_y = min_y + position_frac * height

        dx, dy = cos(angle_rad), sin(angle_rad)
        max_distance = hypot(width, height) * 2

        x1 = ref_x - max_distance * dx
        y1 = ref_y - max_distance * dy
        x2 = ref_x + max_distance * dx
        y2 = ref_y + max_distance * dy

        return LineString([(x1, y1), (x2, y2)])

    def _try_cut_at_position(
        self, position_frac: float, piece: Polygon, angle_rad: float
    ) -> list[CutResult] | None:
        """Attempt to make a cut at the given angle and position."""
        line = self._generate_line_from_angle_pos(piece, angle_rad, position_frac)
        intersections = intersection(line, piece)
        if intersections.is_empty:
            return None

        results: List[CutResult] = []
        geometries = (
            intersections.geoms
            if isinstance(intersections, MultiLineString)
            else [intersections]
        )
        for geom in geometries:
            if isinstance(geom, LineString):
                cut_res = self._validate_and_make_cut(
                    Point(geom.coords[0]), Point(geom.coords[-1]), piece
                )
                if cut_res:
                    results.append(cut_res)
        return results if results else None

    def _golden_section_search_cut(
        self,
        piece: Polygon,
        n_children: int,
        angle_rad: float,
        epsilon: float = 1e-4,
        max_iterations: int = 80,
    ) -> tuple[CutResult | None, RES_TUPLE]:
        """Golden Section Search for the optimal cut position (0.01 to 0.99) at a fixed angle."""

        def evaluate_position(pos: float) -> tuple[CutResult | None, RES_TUPLE]:
            cuts = self._try_cut_at_position(pos, piece, angle_rad)
            best_cut: CutResult | None = None
            best_score: RES_TUPLE = (float("inf"), float("inf"), float("inf"))
            if cuts:
                for cut in cuts:
                    score = self._score_cut_by_division(cut, n_children)
                    if score < best_score:
                        best_score, best_cut = score, cut
            return best_cut, best_score

        a, b = 0.01, 0.99
        c_val = b - (b - a) / PHI
        d_val = a + (b - a) / PHI

        cut_c, score_c = evaluate_position(c_val)
        cut_d, score_d = evaluate_position(d_val)
        best_cut, best_score = min(
            [(cut_c, score_c), (cut_d, score_d)], key=lambda x: x[1]
        )

        for _ in range(max_iterations):
            if abs(b - a) < epsilon:
                break

            if score_c < score_d:
                b = d_val
                d_val = c_val
                c_val = b - (b - a) / PHI
                cut_d, score_d = cut_c, score_c
                cut_c, score_c = evaluate_position(c_val)
            else:
                a = c_val
                c_val = d_val
                d_val = a + (b - a) / PHI
                cut_c, score_c = cut_d, score_d
                cut_d, score_d = evaluate_position(d_val)

            current_best = min([(cut_c, score_c), (cut_d, score_d)], key=lambda x: x[1])
            if current_best[1] < best_score:
                best_cut, best_score = current_best

        return best_cut, best_score

    def _evaluate_angle_gss(
        self, angle_rad: float, piece: Polygon, n_children: int
    ) -> tuple[CutResult | None, RES_TUPLE]:
        """Helper to evaluate a single angle using Golden Section Search for parallel processing."""
        return self._golden_section_search_cut(piece, n_children, angle_rad)

    # --- Core planner on a provided Cake (clone or real) ---

    def _plan_on_cake(self, cake_obj: Cake) -> list[tuple[Point, Point]]:
        """Run divide-and-conquer on a specific Cake object and return the cut list."""
        if not cake_obj.exterior_pieces:
            return []
        return self._divide_and_conquer_with_cake(
            cake_obj.exterior_pieces[0], self.children, cake_obj
        )

    def _divide_and_conquer_with_cake(
        self, piece: Polygon, n_children_to_divide: int, cake_obj: Cake
    ) -> list[tuple[Point, Point]]:
        if n_children_to_divide <= 1:
            return []

        # Coarse Angle Search: 9 angles from 0 to 180 degrees (0 to pi radians)
        angles_coarse = np.linspace(0, pi, 9)

        best_overall_cut: CutResult | None = None
        best_overall_score: RES_TUPLE = (float("inf"), float("inf"), float("inf"))

        # Parallel run of Golden Section Search on all coarse angles
        results_coarse: List[tuple[CutResult | None, RES_TUPLE]] = Parallel(n_jobs=-1)(
            delayed(self._evaluate_angle_gss)(angle, piece, n_children_to_divide)
            for angle in angles_coarse
        )

        for cut, score in results_coarse:
            # Must ensure a valid cut that actually split the piece into two
            if (
                cut is not None
                and score < best_overall_score
                and len(cut.polygons) == 2
            ):
                best_overall_score, best_overall_cut = score, cut

        if not best_overall_cut:
            return []

        # Execute the cut on this cake
        cut_points = best_overall_cut.points
        cake_obj.cut(cut_points[0], cut_points[1])

        # Sort new pieces by area
        p1, p2 = sorted(best_overall_cut.polygons, key=lambda p: p.area)

        # Distribute remaining children: small piece gets floor(N/2), large piece gets ceil(N/2)
        n_small = floor(n_children_to_divide / 2)
        n_large = ceil(n_children_to_divide / 2)

        # Recurse
        small_result = self._divide_and_conquer_with_cake(p1, n_small, cake_obj)
        large_result = self._divide_and_conquer_with_cake(p2, n_large, cake_obj)

        # The final result is the current cut + recursive cuts
        return [cut_points] + small_result + large_result

    # --- Your requested sweep over threshold values (0.001 → 0.245 step 0.002) ---

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """
        Sweep threshold (flat_tol) from 0.001 to 0.245 (step=0.002).
        For each threshold:
          - run full cut pipeline on a cloned cake,
          - compute size_span = max(area) - min(area),
          - compute ratio_score = std dev of per-slice ratios via engine API,
          - print metrics,
        Pick the threshold minimizing (size_span + ratio_score),
        tie-break with smaller ratio_score, then smaller size_span.
        Replay the best threshold on the real cake and return its cuts.
        """
        thresholds = [round(x, 3) for x in np.arange(0.001, 0.245 + 1e-9, 0.01)]

        best_thresh = None
        best_metric_sum = float("inf")
        best_ratio_score = float("inf")
        best_size_span = float("inf")
        best_cuts: list[tuple[Point, Point]] = []

        print(
            "\n[Player1] threshold sweep (flat_tol): choose MIN (size_span + ratio_score)"
        )
        print("  thr     count   size_span     ratio_score")

        for thr in thresholds:
            cake_sim = self.cake.copy()
            self.flat_tol = thr

            # run full plan on the clone
            cuts_try = self._plan_on_cake(cake_sim)

            # collect final pieces & metrics
            try:
                final_pieces = getattr(cake_sim, "exterior_pieces", [])
                areas = [p.area for p in final_pieces] if final_pieces else []
                count = len(areas)
                if count != self.children or count == 0:
                    print(
                        f"  {thr:0.3f}   {count:5d}   {'nan':>9}   {'nan':>12}  (skipped)"
                    )
                    continue

                size_span = max(areas) - min(areas)
                ratios = [cake_sim.get_piece_ratio(p) for p in final_pieces]
                ratio_score = float(np.std(ratios)) if ratios else float("inf")
                metric_sum = size_span + ratio_score

                print(
                    f"  {thr:0.3f}   {count:5d}   {size_span:9.5f}   {ratio_score:12.5f}"
                )

            except Exception:
                print(f"  {thr:0.3f}   ERROR computing metrics")
                continue

            # select min(size_span + ratio_score), then min ratio_score, then min size_span
            better = False
            if metric_sum < best_metric_sum:
                better = True
            elif math_isclose(metric_sum, best_metric_sum):
                if (ratio_score < best_ratio_score) or (
                    math_isclose(ratio_score, best_ratio_score)
                    and size_span < best_size_span
                ):
                    better = True

            if better:
                best_metric_sum = metric_sum
                best_ratio_score = ratio_score
                best_size_span = size_span
                best_thresh = thr
                best_cuts = cuts_try

        if best_thresh is None or not best_cuts:
            print(
                "\n[Player1] No threshold produced a full set. Running once with thr=0.001."
            )
            self.flat_tol = 0.001
            return self._plan_on_cake(self.cake)

        print(
            f"\n[Player1] Selected thr={best_thresh:.3f}  size_span={best_size_span:.5f}  ratio_score={best_ratio_score:.5f}  (sum={best_metric_sum:.5f})\n"
        )

        # replay best threshold on the REAL cake and return its cuts
        self.flat_tol = best_thresh
        return self._plan_on_cake(self.cake)

    def get_max_piece(self, pieces: list[Polygon]) -> Polygon:
        """Return largest polygon from list"""
        return max(pieces, key=lambda piece: piece.area)
