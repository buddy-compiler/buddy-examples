def add_arguments(parser):
    parser.add_argument(
        "--partition-count",
        type=int,
        default=1,
        help="Split the fused BERT graph into this many subgraphs scheduled on homogeneous cores.",
    )
    parser.add_argument("--sequence-length", type=int, default=32)
