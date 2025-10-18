from itertools import product
import os
import signal
import subprocess
import time
from shapely import Polygon

from src.cake import InvalidCakeException, read_cake

CPU_SECONDS = 5 * 60
# 124 is an error code used for timeouts
# see https://www.man7.org/linux/man-pages/man1/timeout.1.html
TIMEOUT_ERROR_CODE = 124


def get_all_cakes():
    files = []
    for node in os.walk("cakes/tournament"):
        for file in node[2]:
            files.append(node[0] + "/" + file)

    valid_cakes: list[str] = []
    invalid_cakes: list[str] = []
    for file in files:
        try:
            _ = read_cake(file, -1, sandbox=True)
            valid_cakes.append(file)
        except InvalidCakeException:
            invalid_cakes.append(file)

    return valid_cakes, invalid_cakes


def cake_to_children(cake_path: str, piece: int):
    vertices = [
        list(map(float, line.strip().split(",")))
        for line in open(cake_path, "r").readlines()[1:]
    ]

    p = Polygon(vertices)

    return int(p.area // piece)


parameters = [
    {"--import-cake": get_all_cakes()[0]},
    {"--player": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]},
]


def combos_as_args(params: list[dict[str, list[str]]]) -> list[list[str]]:
    keys: list[str] = []
    values: list[list[str]] = []
    for d in params:
        ((k, vs),) = d.items()
        keys.append(k)
        values.append(vs)

    args: list[list[str]] = []
    for piece in [20, 30, 40][::-1]:
        for vals in product(*values):
            argv: list[str] = []
            for k, v in zip(keys, vals):
                argv.extend([k, v])

                if k == "--import-cake":
                    children = cake_to_children(v, piece)
                    argv.extend(["--children", str(children)])

            args.append(argv)

    return sorted(args, key=lambda x: x[:-3])


def run_with_timeout(cmd, timeout_sec: int) -> tuple[int, str, str, float]:
    start = time.time()
    p = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=timeout_sec)
        end = time.time()
        return int(p.returncode), out, err, end - start
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return TIMEOUT_ERROR_CODE, out, err, float(CPU_SECONDS)


def writeline(filename: str, line: str):
    with open(filename, "a") as f:
        f.write(line.strip() + "\n")


def main():
    filename = "results.csv"
    header = "cake_path,children,group,size_span,ratios_stdev,seconds"

    writeline(filename, header)

    for arg in combos_as_args(parameters):
        args = ["uv", "run", "main.py"] + arg
        print(f"##\n{args}", flush=True)

        returncode, _, err, cpu_seconds = run_with_timeout(args, CPU_SECONDS)

        if returncode == 0:
            line = f"{err},{cpu_seconds:.4f}"
        elif returncode == TIMEOUT_ERROR_CODE:
            line = f"{arg[1]},{arg[3]},{arg[5]},{-1},{-1},{CPU_SECONDS:.4f}"
        else:
            print(f"player failed: {err}")
            line = f"{arg[1]},{arg[3]},{arg[5]},{-1},{-1},{-1}"

        writeline(filename, line)
        print(line, flush=True)


if __name__ == "__main__":
    main()
