from pathlib import Path
import tomllib


def add_arguments(parser):
    with Path(__file__).with_name("model.toml").open("rb") as file:
        parameters = tomllib.load(file)["parameters"]
    parser.add_argument("--checkpoint", type=str, default=parameters["checkpoint"])
    parser.add_argument("--revision", type=str, default=parameters["revision"])
    parser.add_argument("--sequence-length", type=int, default=parameters["sequence_length"])
    parser.add_argument("--options", type=int, default=parameters["options"])
