from shapely import MultiPolygon, wkb
from shapely.geometry import Polygon, JOIN_STYLE, LineString, Point, MultiPoint
from shapely.validation import explain_validity
from shapely.ops import split
from math import atan2, pi, hypot
from typing import cast
from tkinter import Canvas
import random

from argparse import ArgumentParser
from dataclasses import dataclass
import os
import pathlib
from time import time


@dataclass
class Args:
    gui: bool
    player: int
    import_cake: str | None
    seed: int
    children: int
    export_cake: str | None
    debug: bool
    sandbox: bool


def get_cake_dir():
    return pathlib.Path(os.path.curdir + "/cakes/").resolve()


def sanitize_import_cake(org_import_cake: str | None) -> str | None:
    if org_import_cake is None:
        return org_import_cake

    cake_path = pathlib.Path(org_import_cake).resolve()

    cake_dir = get_cake_dir()

    if not cake_path.is_file():
        raise Exception(f'file with path "{cake_path}" not found')

    try:
        cake_path.relative_to(cake_dir)
    except ValueError:
        raise Exception('provided cake path file must be inside "cakes/" directory')

    return str(cake_path)


def sanitize_export_cake(org_export_cake: str | None) -> str | None:
    if org_export_cake is None:
        return org_export_cake

    cake_path = pathlib.Path(org_export_cake).resolve()

    cake_dir = get_cake_dir()

    if cake_path.exists():
        raise Exception(
            f"Can't export cake to '{org_export_cake}', path already in use."
        )

    try:
        cake_path.relative_to(cake_dir)
    except ValueError:
        raise Exception('provided cake path file must be inside "cakes/" directory')

    return str(cake_path)


def sanitize_seed(org_seed: None | str) -> int:
    if org_seed is None:
        seed = int(time() * 100_000) % 1_000_000
        print(f"Generated seed: {seed}")
        return seed

    return int(org_seed)


def sanitize_player(org_player: str) -> int:
    if org_player.isdigit() and 1 <= int(org_player) <= 10:
        return int(org_player)

    elif org_player == "r":
        return 0

    raise Exception(
        f'unknown `--player` value provided: "{org_player}". Expected digit 1<=10 or "r"'
    )


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--gui", "-g", action="store_true", help="render GUI")
    parser.add_argument(
        "--seed", "-s", type=int, help="Seed used by random number generator"
    )
    parser.add_argument(
        "--player",
        "-p",
        default="r",
        help="Specify which player to run",
    )
    parser.add_argument(
        "--children",
        "-n",
        type=int,
        help="Number of children to serve cake to",
        default=10,
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Display debug info")
    parser.add_argument(
        "--sandbox",
        "-x",
        action="store_true",
        help="Load cakes in sandbox environment. Implies `--gui` flag.",
    )

    # users cannot import and export a cake simultaneously
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--import-cake", "-i", help="path to a cake file within cakes/")
    group.add_argument("--export-cake", "-e", help="path to save generated cake to")

    namespace = parser.parse_args()

    seed = sanitize_seed(namespace.seed)
    player = sanitize_player(namespace.player)
    import_cake = sanitize_import_cake(namespace.import_cake)
    export_cake = sanitize_export_cake(namespace.export_cake)

    args = Args(
        gui=namespace.gui or namespace.sandbox,
        player=player,
        import_cake=import_cake,
        seed=seed,
        children=namespace.children,
        export_cake=export_cake,
        debug=namespace.debug,
        sandbox=namespace.sandbox,
    )

    return args


# CANVAS
CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 1000
CAKE_PORTION = 2 / 3
INFO_PORTION = 1 - CAKE_PORTION
AREA_GAMMA = 6
RATIO_GAMMA = 3
FONT_SIZE = 22
SMALL_FONT_SIZE = 14
CAKE_SCALE = 30  # pixels per cm

## COLORS
CANVAS_BG = "#FCF1E3"
CAKE_CRUST = "#FDAE53"
CAKE_INTERIOR = "#FEDE8C"

# LOGIC
MIN_CAKE_ANGLE_DEGREE = 15
MIN_CAKE_INTERIOR_RATIO = 5 / 10
CRUST_SIZE = 1
MIN_PIECE_AREA = 10

MIN_PIECE_AREA_PER_CHILD = 10
MAX_PIECE_AREA_PER_CHILD = 50

# CONSTANTS
TOL = 1e-5
PIECE_SPAN_TOL = 0.5

TOURNAMENT = True



class InvalidCakeException(Exception):
    pass


def extend_line(line: LineString) -> LineString:
    fraction = 0.05
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


def copy_geom(g):
    return wkb.loads(wkb.dumps(g))


class Cake:
    def __init__(self, p: Polygon, num_children: int, sandbox: bool) -> None:
        self.exterior_shape = p
        self.interior_shape = generate_interior(p)
        self.exterior_pieces = [p]
        self.sandbox = sandbox

        assert_cake_is_valid(
            self.exterior_shape, self.interior_shape, num_children, self.sandbox
        )

    def copy(self):
        new = object.__new__(Cake)
        new.exterior_shape = copy_geom(self.exterior_shape)
        new.interior_shape = copy_geom(self.interior_shape)
        new.exterior_pieces = [copy_geom(p) for p in self.exterior_pieces]

        return new

    def get_piece_sizes(self):
        return [p.area for p in self.exterior_pieces]

    def get_area(self):
        return self.exterior_shape.area

    def get_piece_ratio(self, piece: Polygon):
        if piece.intersects(self.interior_shape):
            inter = piece.intersection(self.interior_shape)
            return inter.area / piece.area if not inter.is_empty else 0
        return 0

    def get_piece_ratios(self):
        ratios = []
        for piece in self.exterior_pieces:
            ratios.append(self.get_piece_ratio(piece))
        return ratios

    def get_offsets(self):
        minx, miny, maxx, maxy = self.exterior_shape.bounds
        x_center = (maxx + minx) * CAKE_SCALE / 2
        y_center = (maxy + miny) * CAKE_SCALE / 2
        x_offset = CANVAS_WIDTH * CAKE_PORTION / 2 - x_center
        y_offset = CANVAS_HEIGHT / 2 - y_center

        return x_offset, y_offset

    def get_scaled_vertex_points(self):
        x_offset, y_offset = self.get_offsets()
        xys = self.get_boundary_points()

        ext_points = []
        for xy in xys:
            x, y = xy.coords[0]
            ext_points.extend(
                [x * CAKE_SCALE + x_offset, y * CAKE_SCALE + y_offset]
            )

        int_xys = self.get_interior_points()

        int_points = []
        for int_xy in int_xys:
            ip = []
            for xy in int_xy:
                x, y = xy.coords[0]
                ip.extend([x * CAKE_SCALE + x_offset, y * CAKE_SCALE + y_offset])
            int_points.append(ip)

        return ext_points, int_points

    def draw(self, canvas: Canvas, draw_angles=False):
        x_offset, y_offset = self.get_offsets()
        ext_coords, int_coords = self.get_scaled_vertex_points()
        # draw crust
        canvas.create_polygon(ext_coords, outline="black", fill=CAKE_CRUST, width=2)

        # draw interiors
        for ic in int_coords:
            canvas.create_polygon(
                ic, outline=CAKE_INTERIOR, fill=CAKE_INTERIOR, width=0
            )

        # write edge points to canvas
        coords = [i.coords[0] for i in self.get_boundary_points()][:-1]
        if draw_angles:
            for idx, (x, y) in enumerate(coords):
                x_str = f"{int(x)}" if x == int(x) else f"{x:.1f}"
                y_str = f"{int(y)}" if y == int(y) else f"{y:.1f}"
                canvas.create_text(
                    (x * CAKE_SCALE) + 50 + x_offset,
                    (y * CAKE_SCALE) - 15 + y_offset,
                    text=f"{idx}: {x_str}, {y_str}",
                    font=("Arial", FONT_SIZE),
                    fill="black",
                    activefill="gray",
                )

    def point_lies_on_piece_boundary(self, p: Point, piece: Polygon):
        return p.distance(piece.boundary) <= TOL

    def get_intersecting_pieces_from_point(self, p: Point):
        touched_pieces = [
            piece
            for piece in self.exterior_pieces
            if self.point_lies_on_piece_boundary(p, piece)
        ]

        return touched_pieces

    def __cut_is_within_cake(self, cut: LineString) -> bool:
        outside = cut.difference(self.exterior_shape.buffer(TOL * 2))
        return outside.is_empty

    def get_cuttable_piece(self, from_p: Point, to_p: Point):
        a_pieces = self.get_intersecting_pieces_from_point(from_p)
        b_pieces = self.get_intersecting_pieces_from_point(to_p)

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

    def __cut_is_valid(
        self, from_p: Point, to_p: Point
    ) -> tuple[bool, str, Polygon | None]:
        """Check whether a cut from `from_p` to `to_p` is valid.

        If invalid, the method returns the reason as the second argument.
        """
        line = LineString([from_p, to_p])

        if not self.__cut_is_within_cake(line):
            return False, "cut is not within cake", None

        cuttable_piece, reason = self.get_cuttable_piece(from_p, to_p)

        if not cuttable_piece:
            return False, reason, None

        return True, "valid", cuttable_piece

    def cut_is_valid(self, from_p: Point, to_p: Point) -> tuple[bool, str]:
        """Check whether a cut from `from_p` to `to_p` is valid. For public use to not cause breaking changes

        If invalid, the method returns the reason as the second argument.
        """
        is_valid, reason, _target_piece = self.__cut_is_valid(from_p, to_p)
        return is_valid, reason

    def does_line_cut_piece_well(self, line: LineString, piece: Polygon):
        """Checks whether line cuts piece in two valid (large enough) pieces"""
        if piece.touches(line):
            return False, "cut lies on piece boundary"

        if not line.crosses(piece):
            return False, "line does not cut through piece"

        cut_pieces = split(piece, line)
        if len(cut_pieces.geoms) != 2:
            return False, f"line cuts piece in {len(cut_pieces.geoms)}, not 2"

        all_sizes_are_good = all([p.area >= MIN_PIECE_AREA for p in cut_pieces.geoms])

        if not all_sizes_are_good:
            return False, "line cuts a piece that's too small"

        return True, ""

    def cut_piece(self, piece: Polygon, from_p: Point, to_p: Point):
        bound = piece.boundary
        a = bound.interpolate(bound.project(from_p))
        b = bound.interpolate(bound.project(to_p))

        line = LineString([a, b])
        # ensure that the line extends beyond the piece
        line = extend_line(line)

        split_piece = split(piece, line)

        split_pieces: list[Polygon] = [
            cast(Polygon, geom) for geom in split_piece.geoms
        ]

        return split_pieces

    def cut(self, from_p: Point, to_p: Point):
        """Perform a cut from `from_p` to `to_p` on this cake."""
        is_valid, reason, target_piece = self.__cut_is_valid(from_p, to_p)
        if not is_valid:
            raise Exception(f"invalid cut: {reason}")

        # as this cut is valid, we will have exactly one cuttable piece
        assert target_piece is not None

        split_pieces = self.cut_piece(target_piece, from_p, to_p)

        # swap out old piece with the two smaller pieces we just cut from it
        target_idx = self.exterior_pieces.index(target_piece)
        self.exterior_pieces.pop(target_idx)
        self.exterior_pieces.extend(split_pieces)

    def get_boundary_points(self) -> list[Point]:
        """Get a list of all boundary points in a (crust, interior) tuple."""
        return [Point(c) for c in self.exterior_shape.exterior.coords]

    def get_interior_points(self) -> list[list[Point]]:
        int_points = []

        if isinstance(self.interior_shape, Polygon):
            i = [Point(c) for c in self.interior_shape.exterior.coords]
            int_points.append(i)

        elif isinstance(self.interior_shape, MultiPolygon):
            for geom in self.interior_shape.geoms:
                i = [Point(c) for c in geom.exterior.coords]
                int_points.append(i)

        return int_points

    def get_pieces(self):
        return self.exterior_pieces

    def pieces_are_even(self):
        areas = [p.area for p in self.exterior_pieces]
        return max(areas) - min(areas) <= PIECE_SPAN_TOL

    def get_angles(self):
        return get_polygon_angles(self.exterior_shape)


def polygon_orientation(points: list[tuple[float, ...]]):
    # >0 for CCW, <0 for CW (shoelace)
    area2 = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return 1 if area2 > 0 else -1  # CCW = +1


def get_polygon_angles(p: Polygon) -> list[float]:
    angles = []
    vertices = list(p.exterior.coords[:-1])

    orient = polygon_orientation(vertices)

    for i, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(i - 1) % len(vertices)]
        x3, y3 = vertices[(i + 1) % len(vertices)]

        ax, ay = x2 - x1, y2 - y1
        bx, by = x3 - x1, y3 - y1

        if hypot(ax, ay) == 0 or hypot(bx, by) == 0:
            angles.append(float("nan"))

        dot = ax * bx + ay * by
        cross = ax * by - ay * bx
        unsigned = atan2(abs(cross), dot)

        is_reflex = (cross < 0 and orient > 0) or (cross > 0 and orient < 0)

        angle = 2 * pi - unsigned if not is_reflex else unsigned
        angles.append(angle * 180 / pi)

    return angles


def cake_angles_are_ok(cake: Polygon):
    return all([angle >= MIN_CAKE_ANGLE_DEGREE for angle in get_polygon_angles(cake)])


def cake_is_ok(cake: Polygon | MultiPolygon) -> tuple[bool, str]:
    if not cake.is_valid:
        return False, explain_validity(cake)
    if cake.is_empty:
        return False, "cake is empty"
    if not cake.area > 0:
        return False, "area <= 0"

    return True, ""


def generate_interior(exterior: Polygon) -> Polygon | MultiPolygon:
    return exterior.buffer(-CRUST_SIZE, join_style=JOIN_STYLE.mitre)


def assert_cake_is_valid(
    cake: Polygon, interior: Polygon | MultiPolygon, num_children: int, sandbox: bool
):
    ok, reason = cake_is_ok(cake)
    if not ok:
        raise InvalidCakeException(f"cake is invalid: {reason}")

    ok, reason = cake_is_ok(interior)
    if not ok:
        raise InvalidCakeException(f"interior is invalid: {reason}")

    # if we're running in a sandbox, we don't care about the additional constraints
    if sandbox:
        return

    if not cake_angles_are_ok(cake):
        raise InvalidCakeException(
            f"Cake has at least one angle < {MIN_CAKE_ANGLE_DEGREE} degrees",
        )

    interior_ratio = interior.area / cake.area
    if interior_ratio < MIN_CAKE_INTERIOR_RATIO:
        raise InvalidCakeException(
            f"cake has too much crust, got {interior_ratio * 100:.1f}%, expected >={MIN_CAKE_INTERIOR_RATIO * 100:.0f}% interior"
        )

    if not (
        MIN_PIECE_AREA_PER_CHILD
        <= cake.area / num_children
        <= MAX_PIECE_AREA_PER_CHILD
    ):
        raise InvalidCakeException(
            f"cake area not between {MIN_PIECE_AREA_PER_CHILD}cm^2 - {MAX_PIECE_AREA_PER_CHILD}cm^2 per child: got {cake.area / num_children:.1f}cm^2"
        )


def read_cake(cake_path: str, num_children: int, sandbox: bool) -> Cake:
    vertices = [
        list(map(float, line.strip().split(",")))
        for line in open(cake_path, "r").readlines()[1:]
    ]

    return Cake(Polygon(vertices), num_children, sandbox)


def write_cake(cake_path: str, cake: Cake):
    vertices = cake.get_boundary_points()

    with open(cake_path, "w") as f:
        f.write("x,y\n")
        f.writelines([f"{v.x},{v.y}\n" for v in vertices])

    print(f"wrote generated cake to '{cake_path}'")


def attempt_cake_generation(num_vertices: int) -> Polygon:
    vertices = []
    for _ in range(num_vertices):
        x = random.randint(0, 30)
        y = random.randint(0, 30)
        vertices.append((x, y))

    # generates a convex, "simple" cake
    hull = MultiPoint(vertices).convex_hull

    if hull.geom_type != "Polygon":
        raise Exception("failed to generate a polygon cake")

    p = cast(Polygon, hull)

    vertices = list(p.exterior.coords)

    while not cake_angles_are_ok(p) and len(p.exterior.coords) - 1 > 3:
        angles = get_polygon_angles(p)
        for i, angle in enumerate(angles):
            if angle < MIN_CAKE_ANGLE_DEGREE:
                vertices.pop(i)
                p = Polygon(vertices)
                break
    return p


def generate_cake(children: int, sandbox: bool) -> Cake:
    # the minimum amount of vertices for the cake polygon
    lo = max(3, int(children / 2))
    # and the maximum..
    hi = max(lo * 2, children * 2)
    num_vertices = random.randint(lo, hi)

    # how often will we try generating a valid cake until we give up
    attempts = 500
    for _ in range(attempts):
        try:
            attempted_cake = attempt_cake_generation(num_vertices)
            cake = Cake(attempted_cake, children, sandbox)

            print(
                f"Generated cake with {len(attempted_cake.exterior.coords) - 1} vertices"
            )
            return cake
        except InvalidCakeException:
            # assert_cake_is_valid failed, try again
            pass

    raise Exception("gave up trying to generate cake")


def cake_from_args(args: Args) -> Cake:
    if args.import_cake:
        return read_cake(args.import_cake, args.children, args.sandbox)
    gen_cake = generate_cake(args.children, args.sandbox)

    if args.export_cake:
        write_cake(args.export_cake, gen_cake)

    return gen_cake
