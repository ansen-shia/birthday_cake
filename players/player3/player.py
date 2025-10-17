from shapely import Point

from players.player import Player
from src.cake import Cake
from .helper_func import find_valid_cuts_binary_search

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy


class Player3(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.num_samples = 500
        self.original_ratio = cake.get_piece_ratio(cake.get_pieces()[0])
        self.target_area = sum(p.area for p in self.cake.get_pieces()) / self.children

    def get_cuts(self) -> list[tuple[Point, Point]]:
        """Greedily generate cuts to divide cake into equal pieces."""
        return self._bucket_cutting_strategy()

    def _greedy_cutting_strategy(self) -> list[tuple[Point, Point]]:
        """Main greedy cutting algorithm."""
        cuts = []
        working_cake = self.cake.copy()
        remaining_children = self.children

        while remaining_children > 1:
            largest_piece = max(working_cake.get_pieces(), key=lambda piece: piece.area)
            # If this is the first cut, let's try to cut in as close to half as possible
            if remaining_children == self.children:
                potential_ratios = [
                    i / remaining_children
                    for i in range(1, (remaining_children // 2) + 1)
                ]
                target_ratio = max(potential_ratios)
            else:
                # Cut one piece at a time
                target_ratio = self.target_area / largest_piece.area

            best_cut = self._find_best_cut_for_piece(
                working_cake, largest_piece, target_ratio
            )

            if best_cut is None:
                break

            cuts.append(best_cut)
            working_cake.cut(best_cut[0], best_cut[1])
            remaining_children -= 1

        return cuts

    def _bucket_cutting_strategy(self) -> list[tuple[Point, Point]]:
        """Main bucket cutting algorithm using recursive splitting (area fairness only)."""
        cuts = []
        largest_piece = max(self.cake.get_pieces(), key=lambda piece: piece.area)
        self._recursive_bucket_cut(self.cake, largest_piece, self.children, cuts)
        return cuts

    def _recursive_cut_worker(self, full_cake, cake_piece, num_children):
        cuts = []
        self._recursive_bucket_cut(
            copy.deepcopy(full_cake), cake_piece, num_children, cuts
        )
        return cuts

    def _evaluate_ratio_worker(self, ratio, working_cake, cake_piece):
        """Runs _find_best_cut_for_piece in parallel for a given ratio."""
        local_cake = copy.deepcopy(working_cake)
        cut = self._find_best_cut_for_piece(local_cake, cake_piece, ratio)
        if cut is None:
            return {
                "ratio": ratio,
                "cut": None,
                "ratio_diff": float("inf"),
                "split": None,
            }

        split_pieces = local_cake.cut_piece(cake_piece, cut[0], cut[1])
        if len(split_pieces) != 2:
            return {
                "ratio": ratio,
                "cut": cut,
                "ratio_diff": float("inf"),
                "split": None,
            }

        ratios = [local_cake.get_piece_ratio(piece) for piece in split_pieces]
        ratio_diffs = [abs(ratio - self.original_ratio) for ratio in ratios]
        ratio_diff = min(ratio_diffs)

        return {
            "ratio": ratio,
            "cut": cut,
            "ratio_diff": ratio_diff,
            "split": split_pieces,
        }

    def _recursive_bucket_cut(
        self,
        full_cake,
        cake_piece,
        num_children: int,
        cuts: list[tuple["Point", "Point"]],
    ):
        """Recursively divide a cake piece for a given number of children based on area fairness."""
        if num_children <= 1:
            return

        working_cake = full_cake.copy()
        potential_ratios = [i / num_children for i in range(1, (num_children // 2) + 1)]
        potential_ratios.sort(reverse=True)
        best_cut = None
        best_ratio_diff = float("inf")
        best_split = None
        # best_ratio = None

        # ---------- PARALLEL RATIO EVALUATION ----------
        results = []

        with ProcessPoolExecutor(max_workers=min(4, len(potential_ratios))) as executor:
            futures = [
                executor.submit(
                    self._evaluate_ratio_worker, ratio, working_cake, cake_piece
                )
                for ratio in potential_ratios
            ]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    results.append(result)

        # Pick the best result
        for result in results:
            if result["ratio_diff"] < best_ratio_diff:
                best_ratio_diff = result["ratio_diff"]
                best_cut = result["cut"]
                best_split = result["split"]
                # best_ratio = result["ratio"]

        if best_cut is None or best_split is None:
            print("No valid cut found for any ratio, stopping.")
            return

        # ---------- RECURSIVE SPLITTING ----------

        left_children = int(round(best_split[0].area / self.target_area))
        right_children = num_children - left_children

        cuts.append(best_cut)
        working_cake.cut(best_cut[0], best_cut[1])

        # ---------- PARALLEL RECURSIVE CUTTING ----------

        if left_children > 1 or right_children > 1:
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = []
                if left_children > 1:
                    futures.append(
                        executor.submit(
                            self._recursive_cut_worker,
                            working_cake,
                            best_split[0],
                            left_children,
                        )
                    )
                if right_children > 1:
                    futures.append(
                        executor.submit(
                            self._recursive_cut_worker,
                            working_cake,
                            best_split[1],
                            right_children,
                        )
                    )

                for f in as_completed(futures):
                    cuts.extend(f.result())

    def _find_best_cut_for_piece(
        self, cake: Cake, piece, desired_cut_ratio: float
    ) -> tuple[Point, Point] | None:
        """Find the best cut for a specific piece using binary search."""
        perimeter_points = self._get_perimeter_points_for_piece(piece)

        valid_cuts = find_valid_cuts_binary_search(
            cake,
            perimeter_points,
            # self.target_area * self.children * desired_cut_ratio,
            piece.area * desired_cut_ratio,
            self.original_ratio,
            piece,
        )

        if not valid_cuts:
            return None

        return valid_cuts[0]

    def _get_perimeter_points_for_piece(self, piece) -> list[Point]:
        """Get perimeter points for a specific piece."""
        boundary = piece.exterior
        points = [
            boundary.interpolate(i / self.num_samples, normalized=True)
            for i in range(self.num_samples)
        ]
        return points
