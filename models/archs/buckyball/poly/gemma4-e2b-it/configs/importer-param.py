from pathlib import Path
from importlib import import_module
import tomllib


def add_arguments(parser):
    model = import_module("stack.models.models.gemma4-e2b-it")
    with Path(model.__file__).parent.joinpath("configs/model.toml").open("rb") as file:
        parameters = tomllib.load(file)["parameters"]
    parser.add_argument("--checkpoint", type=str, default=parameters["checkpoint"])
