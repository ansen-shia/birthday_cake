from shapely import Point, Polygon
from shapely.geometry import LineString
from src.cake import Cake
import math

# essentially our outer circle idea from wednesday, does not yet take into account the ratio idea, but binary search
# should only change minimally


def find_point(xy1: Point, angle: float, piece: Polygon) -> Point | None:
    # Find where a line from start_point at given angle intersects the piece boundary

    # create a line extending from xy1 at the given angle
    # 100 is arbitraryly large to ensure it goes beyond the piece
    xy2 = Point(xy1.x + 100 * math.cos(angle), xy1.y + 100 * math.sin(angle))
    line = LineString([xy1, xy2])

    boundary = piece.boundary
    intersection = line.intersection(boundary)

    if intersection.geom_type == "Point":
        return intersection

    elif intersection.geom_type == "MultiPoint":
        valid_points = (p for p in intersection.geoms if p.distance(xy1) > 0)

        return min(valid_points, key=lambda p: p.distance(xy1))


def binary_search_cut(
    cake: Cake, xy1: Point, target_area: float, largest_piece: Polygon
) -> tuple[Point, float] | None:
    # Use binary search to find a cut that produces the target area

    centroid = largest_piece.centroid

    # calculate angle from xy1 through centroid
    rise = centroid.y - xy1.y
    run = centroid.x - xy1.x
    angle = math.atan2(rise, run)

    xy2 = find_point(xy1, angle, largest_piece)

    if xy2 is None:
        return None
    # calculate initial angle with initial xy2
    rise = xy2.y - xy1.y
    run = xy2.x - xy1.x
    xy2_angle = math.atan2(rise, run)

    # we want to search +/- 180 degrees from initial angle
    left = xy2_angle - 2 * math.pi
    right = xy2_angle + 2 * math.pi

    best_xy2 = None
    best_area_diff = float("inf")

    # standard binary search. stop when within 0.001 radians
    while right > 0.001 + left:
        mid = (left + right) / 2
        xy2 = find_point(xy1, mid, largest_piece)

        if xy2 is None:
            left = mid
            continue

        valid, _ = cake.cut_is_valid(xy1, xy2)

        if not valid:
            left = mid
            continue

        target_piece, _ = cake.get_cuttable_piece(xy1, xy2)

        pieces = cake.cut_piece(target_piece, xy1, xy2)
        areas = [piece.area for piece in pieces]

        # check area
        area_diffs = [abs(area - target_area) for area in areas]
        area_diff = min(area_diffs)

        # save best cut as of now
        if area_diff < best_area_diff:
            best_area_diff = area_diff
            best_xy2 = xy2

        min_area = min(areas)

        if min_area < target_area:
            left = mid
        else:
            right = mid

    # reduce threshold
    if best_xy2 and best_area_diff < 0.20:
        return (best_xy2, best_area_diff)

    return None


def find_valid_cuts_binary_search(
    cake: Cake,
    perim_points: list[Point] | None,
    target_area: float,
    original_ratio: float,
    largest_piece: Polygon,
) -> list[tuple[Point, Point]]:
    valid_cuts = []

    # largest_piece = max(cake.get_pieces(), key=lambda piece: piece.area)

    for xy1 in perim_points:
        cut = binary_search_cut(cake, xy1, target_area, largest_piece)

        if not cut:
            continue

        xy2, area_diff = cut

        split_pieces = cake.cut_piece(largest_piece, xy1, xy2)
        ratios = [cake.get_piece_ratio(piece) for piece in split_pieces]
        ratio_diffs = [abs(ratio - original_ratio) for ratio in ratios]

        # find the worst ratio diff since both pieces should conserve ratio
        ratio_diff = max(ratio_diffs)

        valid_cuts.append((xy1, xy2, area_diff, ratio_diff))

    # sort only by ratio since we only return valid areas
    valid_cuts.sort(key=lambda x: x[3])

    return [(xy1, xy2) for xy1, xy2, _, _ in valid_cuts]
