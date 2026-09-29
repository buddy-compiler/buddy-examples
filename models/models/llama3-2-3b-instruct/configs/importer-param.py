def add_arguments(parser):
    parser.add_argument("--max-cache-len", type=int, default=16)
    parser.add_argument("--prefill-len", type=int, default=8)
    parser.add_argument("--parts", type=int, default=4)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
