from starganv2.options import build_parser
from starganv2.core.solver import Solver


def test_solver_init():
    parser = build_parser()
    args = parser.parse_args(["--mode", "train"])
    assert args.mode == "train"
    solver = Solver(args)
