from dataclasses import dataclass
from players.player import Player
from src.cake import Cake
from shapely.geometry import Polygon, LineString, Point
from shapely import intersection
from shapely.ops import split
from typing import cast
from math import hypot

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

    def find_common_points(self, pieces: list[Polygon]) -> tuple[tuple[float, float]]:
        """Find the vertices of the new cake polygon once cut"""
        return tuple(set(pieces[0].exterior.coords) & set(pieces[1].exterior.coords))

    def __cut_is_within_cake(self, cut: LineString, polygon: Polygon) -> bool:
        outside = cut.difference(polygon.buffer(c.TOL * 2))
        return outside.is_empty
    
    def get_intersecting_pieces_from_point(self, p: Point, polygons: list[Polygon]):
        touched_pieces = [
            piece
            for piece in polygons
            if self.point_lies_on_piece_boundary(p, piece)
        ]

        return touched_pieces
    

    def get_scaled_vertex_points(self):
        x_offset, y_offset = self.get_offsets()
        xys = self.get_boundary_points()

        ext_points = []
        for xy in xys:
            x, y = xy.coords[0]
            ext_points.extend(
                [x * c.CAKE_SCALE + x_offset, y * c.CAKE_SCALE + y_offset]
            )

        int_xys = self.get_interior_points()

        int_points = []
        for int_xy in int_xys:
            ip = []
            for xy in int_xy:
                x, y = xy.coords[0]
                ip.extend([x * c.CAKE_SCALE + x_offset, y * c.CAKE_SCALE + y_offset])
            int_points.append(ip)

        return ext_points, int_points

    def point_lies_on_piece_boundary(self, p: Point, piece: Polygon):
        return p.distance(piece.boundary) <= c.TOL

    def get_cuttable_piece(self, from_p: Point, to_p: Point, polygons: list[Polygon]):
        
        print("get_cuttable_pieces: ", from_p, to_p, polygons)
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

    def cut_is_valid(self, from_p: Point, to_p: Point, polygon: Polygon) -> tuple[bool, str]:
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
    
    def virtual_cut(self, piece: Polygon, cut: LineString) -> CutResult:
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
        #x, y = cut.coords
        coords = list(cut.coords)
        from_p, to_p = Point(coords[0]), Point(coords[1])

        is_valid, reason = self.cut_is_valid(from_p, to_p, piece)
        # NOTE: catch this later
        if not is_valid:
            print("invalid line cut")
            print(reason)
            return None

        # as this cut is valid, we will have exactly one cuttable piece
        target_piece, _ = self.get_cuttable_piece(from_p, to_p, [piece])
        if not target_piece:
            print("invalid piece cut out")
            return None

        bound = piece.boundary
        a = bound.interpolate(bound.project(from_p))
        b = bound.interpolate(bound.project(to_p))

        line = LineString([a, b])
        # ensure that the line extends beyond the piece
        # line = extend_line(line)
        # cut the cake
        split_piece = split(piece, line)
        split_pieces: list[Polygon] = [
            cast(Polygon, geom) for geom in split_piece.geoms
        ]
        #point1, point2 = line.coords
        #output = CutResult(polygons=split_pieces, points=(Point(point1), Point(point2)))
        output = CutResult(polygons=split_pieces, points=(a, b))

        return output
    
    def current_polygon(self, points: tuple[Point, Point], polygon: Polygon):
        res = []
        for x,y in polygon.exterior.coords:
            if Point([x,y]) in points:
                res.append(polygon)
        return res

    
    def score_cut(self, res: CutResult) -> float:
        target_area = self.cake.get_area() / self.children
        polygons = self.current_polygon(res.points, res.polygons)
        if len(polygons) > 2:
            raise "Expected 2 polygons"
        return [abs(polygon.area - target_area) for polygon in polygons]
            

    def get_cuts(self) -> list[tuple[Point, Point]]:
        min_x, min_y, max_x, max_y = self.cake.exterior_shape.bounds
        result: list[tuple[Point, Point]] = []
        largest_piece = self.get_max_piece(self.cake.exterior_pieces) # Generate cuts incrementally
        total_area = self.cake.get_area()
        ideal_area = total_area / self.children
        for i in range(1, self.children):
            # make n-1 cuts
            # cut of piece of size 1/n, then another of size 1/n, etc - swiping along x-axis or y-axis of largest piece
            # Try x-slice first, then y-slice if x fails
            cut_result_1 = self._try_x_slice(i, min_x, max_x, min_y, max_y, largest_piece)
            cut_result_2 = self._try_y_slice(i, min_x, max_x, min_y, max_y, largest_piece)
            
            cuts = [cut_result_1, cut_result_2]

            for cut in cuts:
                min(self.score_cut(cut))
                
            if cut_result:
                largest_piece = self.get_max_piece(cut_result.polygons)
                result.append(cut_result.points)
                print(f"DEBUG: Cut successful, new largest piece area: {largest_piece.area}")
            else:
                print(f"DEBUG: No valid cut found for iteration {i}")
                break
        
        print(f"DEBUG: Generated {len(result)} cuts: {result}")
        return result
    
    def _try_x_slice(self, x_position: float, min_x: float, 
                     min_y: float, max_y: float, child_area: float, piece: Polygon):
        # find leftmost point on the polygon? - it will be a vertice - no matter what - or two vertices if it's a flat edge
        # do vertice with the lowest x and y - because that is always the direction we head in
        # increment by 0.1 with every iteration
        x_slice = LineString([[x_position, min_y], [x_position, max_y]])
        intersection_points = intersection(x_slice, piece)
        point_1 = Point(x_position, min_y)
        point_2 = Point(x_position, max_y)
        if intersection_points.geom_type == 'Point':
            point_1 = intersection_points
            point_2= intersection_points
        elif intersection_points.geom_type == 'LineString':
            point_1 = Point(intersection_points.coords[0][0], intersection_points.coords[0][1])
            point_2= Point(intersection_points.coords[1][0], intersection_points.coords[1][1])
        else:
            # no multiple intersecting sections yet
            return None
        copy_piece = piece.copy()
        bound = copy_piece.boundary
        a = bound.interpolate(bound.project(point_1))
        b = bound.interpolate(bound.project(point_2))

        line = LineString([a, b])
        # ensure that the line extends beyond the piece
        # line = extend_line(line)

        split_piece = split(copy_piece, line)
        print(split_piece)
        split_pieces: list[Polygon] = [
            cast(Polygon, geom) for geom in split_piece.geoms
        ]
        while (split_piece[0].area < child_area - 0.25):
            x_position += 0.01
            x_slice = LineString([[x_position, min_y], [x_position, max_y]])
            intersection_points = intersection(x_slice, piece)
            point_1 = Point(x_position, min_y)
            point_2 = Point(x_position, max_y)
            if intersection_points.geom_type == 'Point':
                point_1 = intersection_points
                point_2= intersection_points
            elif intersection_points.geom_type == 'LineString':
                point_1 = Point(intersection_points.coords[0][0], intersection_points.coords[0][1])
                point_2= Point(intersection_points.coords[1][0], intersection_points.coords[1][1])
            else:
                # no multiple intersecting sections yet
                return None
            copy_piece = piece.copy()
            bound = copy_piece.boundary
            a = bound.interpolate(bound.project(point_1))
            b = bound.interpolate(bound.project(point_2))

        line = LineString([a, b])
        # ensure that the line extends beyond the piece
        # line = extend_line(line)

        split_piece = split(copy_piece, line)
        print(split_piece)
        split_pieces: list[Polygon] = [
            cast(Polygon, geom) for geom in split_piece.geoms
        ]


    
    # def _try_x_slice(self, iteration: int, min_x: float, max_x: float, 
    #                  min_y: float, max_y: float, piece: Polygon) -> CutResult | None:
    #     """Attempt to make a vertical cut at the given iteration."""
    #     x_position = iteration / self.children * (max_x - min_x) + min_x
    #     x_slice = LineString([[x_position, min_y], [x_position, max_y]])
        
    #     print(f"DEBUG: Trying x-slice at x={x_position}")
        
    #     # Find intersection with the piece
    #     intersections = intersection(x_slice, piece)
    #     if isinstance(intersections, LineString) and not intersections.is_empty:
    #         x_slice = intersections
    #         print(f"DEBUG: X-slice intersects piece: {x_slice}")
    #         return self.virtual_cut(piece, x_slice)
    #     else:
    #         print(f"DEBUG: X-slice doesn't intersect piece properly: {intersections}")
    #         return None
    
    def _try_y_slice(self, iteration: int, min_x: float, max_x: float,
                     min_y: float, max_y: float, piece: Polygon) -> CutResult | None:
        """Attempt to make a horizontal cut at the given iteration."""
        y_position = iteration / self.children * (max_y - min_y) + min_y
        y_slice = LineString([[min_x, y_position], [max_x, y_position]])
        
        print(f"DEBUG: Trying y-slice at y={y_position}")
        
        # Find intersection with the piece
        intersections = intersection(y_slice, piece)
        if isinstance(intersections, LineString) and not intersections.is_empty:
            y_slice = intersections
            print(f"DEBUG: Y-slice intersects piece: {y_slice}")
            return self.virtual_cut(piece, y_slice)
        else:
            print(f"DEBUG: Y-slice doesn't intersect piece properly: {intersections}")
            return None

    def get_max_piece(self, pieces):
        return max(pieces, key=lambda piece: piece.area)