from shapely import Point, wkb, Polygon
from shapely.geometry import LineString as LS
from players.player import Player
from src.cake import Cake
import math
from typing import List, Tuple, Optional


def copy_geom(g):
    return wkb.loads(wkb.dumps(g))


class Player5(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        print(f"I am {self}")
        self.target_area = cake.get_area() / children
        total_crust_area = cake.get_area() - cake.interior_shape.area
        self.target_crust_ratio = total_crust_area / cake.get_area()
        self.moves: list[tuple[Point, Point]] = []
        self.max_area_deviation = 0.25
        self.sample_step = 1.0

    def copy_cake(self, cake):
        new = object.__new__(Cake)
        new.exterior_shape = self.cake.exterior_shape
        new.interior_shape = self.cake.interior_shape
        new.exterior_pieces = [copy_geom(p) for p in self.cake.exterior_pieces]
        return new

    def evaluate_cut(self, from_p: Point, to_p: Point) -> float:
        cake_copy = self.copy_cake(self.cake)
        try:
            cake_copy.cut(from_p, to_p)

            for piece in cake_copy.exterior_pieces:
                piece_size = piece.area
                area_multiple = round(piece_size / self.target_area)
                nearest_area_multiple = area_multiple * self.target_area
                if abs(piece_size - nearest_area_multiple) > self.max_area_deviation:
                    return float("inf")

            sumsq = 0.0
            max_dev = 0.0
            for piece in cake_copy.exterior_pieces:
                interior_ratio = cake_copy.get_piece_ratio(piece)
                crust_ratio = 1 - interior_ratio
                d = crust_ratio - self.target_crust_ratio
                k_children = max(1, round(piece.area / self.target_area))
                sumsq += k_children * (d * d)
                if abs(d) > max_dev:
                    max_dev = abs(d)

            return sumsq + 0.5 * max_dev

        except Exception:
            return float("inf")

    def get_sample_points(self, piece: Polygon, step: float = None) -> List[Point]:
        if step is None:
            step = self.sample_step
        coords = list(piece.exterior.coords[:-1])
        raw_points: list[tuple[float, float]] = []
        for move in self.moves:
            for point in move:
                if self.cake.point_lies_on_piece_boundary(point, piece):
                    raw_points.append((point.x, point.y))
        for i in range(len(coords)):
            next_i = (i + 1) % len(coords)
            x1, y1 = coords[i]
            x2, y2 = coords[next_i]
            raw_points.append((x1, y1))
            dx = x2 - x1
            dy = y2 - y1
            length = (dx * dx + dy * dy) ** 0.5
            if step > 0 and length > step * 3:
                k = 1
                while k * step < length:
                    t = (k * step) / length
                    px = x1 + t * dx
                    py = y1 + t * dy
                    raw_points.append((px, py))
                    k += 1
            elif length > 1:
                mx = x1 + 0.5 * dx
                my = y1 + 0.5 * dy
                raw_points.append((mx, my))
        seen = set()
        sample_points: list[Point] = []
        for x, y in raw_points:
            key = (round(x, 6), round(y, 6))
            if key in seen:
                continue
            seen.add(key)
            sample_points.append(Point(x, y))
        return sample_points

    def find_best_cut(self) -> Tuple[Point, Point]:
        pieces = self.cake.get_pieces()
        if not pieces:
            raise Exception("no pieces available to cut")
        piece = max(pieces, key=lambda p: p.area)
        sample_points = self.get_sample_points(piece)
        min_len = 2.0
        candidate_cuts = []
        for i in range(len(sample_points)):
            for j in range(i + 1, len(sample_points)):
                if sample_points[i].distance(sample_points[j]) < min_len:
                    continue
                from_p = sample_points[i]
                to_p = sample_points[j]
                score = self.evaluate_cut(from_p, to_p)
                if score != float("inf"):
                    candidate_cuts.append((score, from_p, to_p))
            candidate_cuts.sort(key=lambda x: x[0])
            candidate_cuts = candidate_cuts[:50]
        if not candidate_cuts:
            raise Exception("could not find a valid cut")
        candidate_cuts.sort(key=lambda x: x[0])
        _, best_from_p, best_to_p = candidate_cuts[0]
        optimized_from_p, optimized_to_p = self.optimize_cut(best_from_p, best_to_p)
        return optimized_from_p, optimized_to_p

    def get_boundary_direction(
        self, piece: Polygon, point: Point
    ) -> Tuple[float, float]:
        boundary = piece.boundary
        closest_point = boundary.interpolate(boundary.project(point))
        distance = boundary.project(closest_point)
        offset = 0.01
        before_point = boundary.interpolate((distance - offset) % boundary.length)
        after_point = boundary.interpolate((distance + offset) % boundary.length)
        dx = after_point.x - before_point.x
        dy = after_point.y - before_point.y
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            return dx / length, dy / length
        else:
            return 0.0, 0.0

    def optimize_cut(
        self, from_p: Point, to_p: Point, iterations: int = 20
    ) -> Tuple[Point, Point]:
        best_cut = (from_p, to_p)
        best_score = self.evaluate_cut(from_p, to_p)
        if best_score == float("inf"):
            return best_cut
        if best_score == 0:
            return best_cut
        cuttable_piece, _ = self.cake.get_cuttable_piece(from_p, to_p)
        if not cuttable_piece:
            return best_cut
        current_from = Point(from_p.x, from_p.y)
        current_to = Point(to_p.x, to_p.y)
        initial_step_size = self.sample_step / 2
        for iteration in range(iterations):
            step_size = initial_step_size * (1 - iteration / iterations)
            improved = False
            for direction in [-1, 1]:
                from_dx, from_dy = self.get_boundary_direction(
                    cuttable_piece, current_from
                )
                new_from = Point(
                    current_from.x + direction * step_size * from_dx,
                    current_from.y + direction * step_size * from_dy,
                )
                new_score = self.evaluate_cut(new_from, current_to)
                if new_score < best_score:
                    best_score = new_score
                    best_cut = (new_from, current_to)
                    current_from = new_from
                    improved = True
                    break
                to_dx, to_dy = self.get_boundary_direction(cuttable_piece, current_to)
                new_to = Point(
                    current_to.x + direction * step_size * to_dx,
                    current_to.y + direction * step_size * to_dy,
                )
                new_score = self.evaluate_cut(current_from, new_to)
                if new_score < best_score:
                    best_score = new_score
                    best_cut = (current_from, new_to)
                    current_to = new_to
                    improved = True
                    break
            if not improved:
                continue
        return best_cut[0], best_cut[1]

    def get_cuts(self) -> List[Tuple[Point, Point]]:
        self.moves.clear()  # main cutting loop
        for cut in range(self.children - 1):
            try:
                from_p, to_p = self.find_best_cut()
                self.moves.append((from_p, to_p))
                self.cake.cut(from_p, to_p)
            except Exception:
                pieces = self.cake.get_pieces()
                if not pieces:
                    break
                piece = max(pieces, key=lambda p: p.area)
                emergency_cut = self.find_emergency_cut(piece)
                if emergency_cut:
                    a, b = emergency_cut
                    self.cake.cut(a, b)
                    self.moves.append((a, b))
        return self.moves

    def find_emergency_cut(self, piece: Polygon) -> Optional[Tuple[Point, Point]]:
        # fallback if no valid cut found
        minx, miny, maxx, maxy = piece.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        for angle in [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]:
            dx, dy = math.cos(angle), math.sin(angle)
            diam = max(1e-6, math.hypot(maxx - minx, maxy - miny))
            line_length = diam * 2
            a_inf = Point(cx - dx * line_length, cy - dy * line_length)
            b_inf = Point(cx + dx * line_length, cy + dy * line_length)
            line = LS([a_inf, b_inf])
            inter = piece.intersection(line)
            if inter.is_empty:
                continue
            chord = None
            if isinstance(inter, LS):
                chord = inter
            elif hasattr(inter, "geoms"):
                lines = [g for g in inter.geoms if isinstance(g, LS)]
                if lines:
                    chord = max(lines, key=lambda L_: L_.length)
            if chord is None:
                continue
            coords = list(chord.coords)
            if len(coords) < 2:
                continue
            a, b = Point(coords[0]), Point(coords[-1])
            is_valid, _ = self.cake.cut_is_valid(a, b)
            if is_valid:
                return (a, b)
        coords = list(piece.exterior.coords[:-1])
        if len(coords) >= 2:
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    a = Point(coords[i])
                    b = Point(coords[j])
                    is_valid, _ = self.cake.cut_is_valid(a, b)
                    if is_valid:
                        return (a, b)
        boundary = piece.boundary
        if boundary.length > 0:
            a = boundary.interpolate(0)
            b = boundary.interpolate(boundary.length / 2)
            is_valid, _ = self.cake.cut_is_valid(a, b)
            if is_valid:
                return (a, b)
        if len(coords) >= 2:
            return (Point(coords[0]), Point(coords[1]))
        else:
            return (Point(0, 0), Point(1, 1))
