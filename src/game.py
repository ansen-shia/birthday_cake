from sys import stderr
import tkinter as tk
from shapely import LineString
import random
from statistics import stdev

from src.args import Args
import src.constants as c
from src.cake import cake_from_args

from players.player import Player, PlayerException
from players.random_player import RandomPlayer
from players.player1.player import Player1
from players.player2.player import Player2
from players.player3.player import Player3
from players.player4.player import Player4
from players.player5.player import Player5
from players.player6.player import Player6
from players.player7.player import Player7
from players.player8.player import Player8
from players.player9.player import Player9
from players.player10.player import Player10


class Game:
    def handle_play(self):
        self.root.after(50, self.play)

    def score_to_color_pow(self, v: float, gamma: float = 2.2) -> str:
        v = max(0.0, min(1.0, v))
        t = v**gamma  # nonlinear progress
        r = int(255 * (1 - t))  # fade red slowly
        g = int(255 * t)  # grow green slowly
        b = 0
        return f"#{r:02x}{g:02x}{b:02x}"

    def print_overlay_message(self, msg: str):
        size_x, size_y = 500, 300
        pos_x = c.CANVAS_WIDTH / 2 - size_x / 2
        pos_y = c.CANVAS_HEIGHT / 2 - size_y / 2
        self.canvas.create_rectangle(
            pos_x, pos_y, pos_x + size_x, pos_y + size_y, fill="white", stipple="gray50"
        )
        self.canvas.create_text(
            pos_x + size_x / 2,
            pos_y + size_y / 2,
            text=msg,
            font=("Arial", c.FONT_SIZE, "bold"),
            fill="red",
            width=size_x * 0.8,
        )

    def create_colored_text(self, canvas, x, y, segments):
        ids = []
        cur_x = x
        for seg in segments:
            item_id = canvas.create_text(
                cur_x,
                y,
                text=seg["text"],
                fill=seg.get("fill", "black"),
                font=seg.get("font", ("Arial", 12)),
                anchor="w",
            )
            ids.append(item_id)
            # advance x by this segment's width
            x0, y0, x1, y1 = canvas.bbox(item_id)
            cur_x = x1
        return ids

    def draw_cut_areas(self):
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area

        if self.pieces:
            for p in self.pieces:
                self.info.delete(p)

        size_score = self.cake.pieces_are_even()

        pieces = []
        for i, p in enumerate(self.cake.get_pieces()):
            piece_ratio = self.cake.get_piece_ratio(p)
            x = 20
            y = 240 + i * 32

            ratio_score = 1 - abs(piece_ratio - target_ratio) / target_ratio

            ids = self.create_colored_text(
                self.info,
                x,
                y,
                [
                    {
                        "text": f"{i}: size=",
                        "fill": "black",
                        "font": ("Arial", c.FONT_SIZE),
                    },
                    {
                        "text": f"{p.area:.2f}cm^2",
                        "fill": self.score_to_color_pow(size_score, gamma=c.AREA_GAMMA),
                        "font": ("Arial", c.FONT_SIZE, "bold"),
                    },
                    {
                        "text": ", ratio=",
                        "fill": "black",
                        "font": ("Arial", c.FONT_SIZE),
                    },
                    {
                        "text": f"{piece_ratio:.2f}",
                        "fill": self.score_to_color_pow(
                            ratio_score, gamma=c.RATIO_GAMMA
                        ),
                        "font": ("Arial", c.FONT_SIZE, "bold"),
                    },
                ],
            )
            pieces.extend(ids)
        self.pieces = pieces

    def create_buttons(self):
        def on_enter(e):
            e.widget["foreground"] = "gray"

        def on_leave(e):
            e.widget["foreground"] = "black"

        btn = tk.Button(
            self.info,
            text="Play",
            font=("Arial", c.FONT_SIZE),
            command=self.handle_play,
        )
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        btn.pack(pady=10)

        self.children_text = self.info.create_text(
            20,
            100,
            text=f"children: {self.args.children}",
            font=("Arial", c.FONT_SIZE),
            fill="black",
            activefill="gray",
            tags="children_text",
            anchor="w",
        )

        target_area = self.cake.exterior_shape.area / self.args.children
        target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area

        self.info.create_text(
            20,
            170,
            text=f"TARGET\nsize={target_area:.2f}cm^2, ratio={target_ratio:.2f}",
            font=("Arial", c.FONT_SIZE),
            fill="black",
            anchor="w",
        )

    def draw_cake(self):
        self.cake.draw(self.canvas, draw_angles=self.args.debug)

    def get_player(self):
        players: list[type[Player]] = [
            RandomPlayer,
            Player1,
            Player2,
            Player3,
            Player4,
            Player5,
            Player6,
            Player7,
            Player8,
            Player9,
            Player10,
        ]
        assert 0 <= self.args.player <= len(players)
        return players[self.args.player]

    def play(self):
        x_offset, y_offset = self.cake.get_offsets()
        moves = []
        try:
            moves = self.player.get_cuts()
        except PlayerException as e:
            msg = f"{str(e)[:100]}"
            if self.args.gui:
                self.print_overlay_message(msg)
            raise e
        except Exception as e:
            msg = f"Exception: {str(e)[:100]}"
            if self.args.gui:
                self.print_overlay_message(msg)
            raise e

        if len(moves) != self.args.children - 1:
            msg = f"Player Exception: Invalid amount of cuts. expected {self.args.children - 1}, got {len(moves)}"
            if self.args.gui:
                self.print_overlay_message(msg)
            raise Exception(msg)

        for from_p, to_p in moves:
            cut = LineString([from_p, to_p])

            (x0, y0), (x1, y1) = list(cut.coords)

            self.cake.cut(from_p, to_p)

            if self.args.gui:
                self.canvas.create_line(
                    x0 * c.CAKE_SCALE + x_offset,
                    y0 * c.CAKE_SCALE + y_offset,
                    x1 * c.CAKE_SCALE + x_offset,
                    y1 * c.CAKE_SCALE + y_offset,
                    width=2,
                    fill="black",
                )

                if self.args.debug:
                    self.canvas.create_text(
                        x0 * c.CAKE_SCALE + x_offset,
                        y0 * c.CAKE_SCALE - 10 + y_offset,
                        text=f"({x0:.1f}, {y0:.1f})",
                        font=("Arial", c.SMALL_FONT_SIZE),
                        fill="black",
                    )
                    self.canvas.create_text(
                        x1 * c.CAKE_SCALE + x_offset,
                        y1 * c.CAKE_SCALE - 10 + y_offset,
                        text=f"({x1:.1f}, {y1:.1f})",
                        font=("Arial", c.SMALL_FONT_SIZE),
                        fill="black",
                    )

                self.draw_cut_areas()

        if len(self.cake.get_pieces()) != self.args.children:
            raise Exception(
                f"Invalid amount of pieces: expected {self.args.children}, got {len(self.cake.get_pieces())}"
            )

        ratios = [r * 100 for r in self.cake.get_piece_ratios()]
        size_span = max(self.cake.get_piece_sizes()) - min(self.cake.get_piece_sizes())
        size_score = self.cake.pieces_are_even()

        ratio_score = stdev(ratios)

        if self.args.gui:
            self.info.create_text(
                20,
                c.CANVAS_HEIGHT - 130,
                text="Score:",
                font=("Arial", c.FONT_SIZE),
                fill="black",
                anchor="w",
            )

            self.create_colored_text(
                self.info,
                20,
                c.CANVAS_HEIGHT - 90,
                [
                    {
                        "text": "size span = ",
                        "fill": "black",
                        "font": ("Arial", c.FONT_SIZE),
                    },
                    {
                        "text": f"{size_span:.2f}cm^2",
                        "fill": self.score_to_color_pow(size_score),
                        "font": ("Arial", c.FONT_SIZE, "bold"),
                    },
                ],
            )
            self.info.create_text(
                20,
                c.CANVAS_HEIGHT - 50,
                text=f"stdev(ratios) = {ratio_score:.2f}",
                font=("Arial", c.FONT_SIZE),
                fill="black",
                anchor="w",
            )

        print(
            f"SCORE:\nsize span: {size_span:.2f}cm^2\nstdev(ratio): {ratio_score:.2f}"
        )


        if c.TOURNAMENT:
            self.args.import_cake = str(self.args.import_cake)
            import_cake = self.args.import_cake[self.args.import_cake.index("cakes") :]
            print(
                f"{import_cake},{self.args.children},{self.args.player},{size_span:.4f},{ratio_score:.4f}",
                file=stderr,
                end="",
            )

    def __init__(self, args: Args):
        random.seed(args.seed)
        self.args = args
        self.cake = cake_from_args(self.args)

        if not self.args.sandbox:
            self.player = (self.get_player())(
                children=self.args.children,
                cake=self.cake.copy(),
                cake_path=self.args.import_cake,
            )

        if self.args.gui:
            self.pieces = None

            self.root = tk.Tk()
            self.root.title("Birthday cake")

            self.left_frame = tk.Frame(self.root, bg=c.CANVAS_BG)
            self.left_frame.pack(side="left", fill="both", expand=True)

            self.canvas = tk.Canvas(
                self.left_frame,
                height=c.CANVAS_HEIGHT,
                width=c.CANVAS_WIDTH * c.CAKE_PORTION,
                bg=c.CANVAS_BG,
                highlightthickness=0,
            )
            self.canvas.pack(fill="both", expand=True)

            if not self.args.sandbox:
                self.right_frame = tk.Frame(
                    self.root, bg="#f3f3f3", width=c.CANVAS_WIDTH * c.INFO_PORTION
                )
                self.right_frame.pack(side="right", fill="y")
                self.right_frame.pack_propagate(False)
                self.info = tk.Canvas(
                    self.right_frame,
                    height=c.CANVAS_HEIGHT,
                    width=c.CANVAS_WIDTH * c.INFO_PORTION,
                    bg=c.CANVAS_BG,
                )
                self.info.pack(fill="both", expand=True, padx=8, pady=8)

                self.create_buttons()
            self.draw_cake()

        if self.args.gui:
            self.root.mainloop()
        elif not self.args.sandbox:
            self.play()
