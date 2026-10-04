from pathlib import Path
import tomllib


def add_arguments(parser):
    with Path(__file__).with_name("model.toml").open("rb") as file:
        parameters = tomllib.load(file)["parameters"]
    parser.add_argument("--weights", type=str, default=parameters["weights"])
