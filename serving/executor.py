import atexit
import json
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Executor:
    def __init__(self, simulator: Path, program: Path, model_dir: Path, log_root: Path,
                 tile_index: int = 0, record_io: bool = False, arguments: tuple[str, ...] = (), memory_mib: int = 3072,
                 transport=None, rank: int = 0):
        log_root.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(tempfile.mkdtemp(prefix=f"tile-{tile_index}-", dir=log_root))
        for kind in ("cycle", "tensor"):
            (self.log_dir / "trace" / kind).mkdir(parents=True)
        self.log = (self.log_dir / "simulator.log").open("wb")
        self.lock = threading.Lock()
        self.record_io = record_io
        self.sequence = 0
        self.tile_index = tile_index
        self.events = (self.log_dir / "requests.jsonl").open("w")
        self.transport = transport
        self.rank = rank
        self.process = None if transport is not None else subprocess.Popen(
            [str(simulator), "--elf", str(program), "--log-dir", str(self.log_dir),
             "--tile-index", str(tile_index),
             "--memory-mib", str(memory_mib),
             "--", str(model_dir), *arguments],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            bufsize=0,
        )
        self.closed = False
        atexit.register(self.close)

    def execute(self, request: bytes, result_bytes: int) -> bytes:
        with self.lock:
            started = time.monotonic_ns()
            if self.record_io:
                (self.log_dir / f"{self.sequence:04d}-input.bin").write_bytes(request)
            if self.transport is not None:
                result = self.transport.execute(self.rank, request, result_bytes)
            else:
                pending = memoryview(request)
                while pending:
                    written = self.process.stdin.write(pending)
                    if not written:
                        raise RuntimeError(f"BEMU input closed: {self.log_dir}")
                    pending = pending[written:]
                result = bytearray()
                while len(result) < result_bytes:
                    chunk = self.process.stdout.read(result_bytes - len(result))
                    if not chunk:
                        raise RuntimeError(f"BEMU exited before completing the request: {self.log_dir}")
                    result.extend(chunk)
            if self.record_io:
                (self.log_dir / f"{self.sequence:04d}-output.bin").write_bytes(result)
            self.events.write(json.dumps({"tile": self.tile_index, "sequence": self.sequence,
                                          "start_ns": started, "end_ns": time.monotonic_ns(),
                                          "request_bytes": len(request), "result_bytes": result_bytes}) + "\n")
            self.events.flush()
            self.sequence += 1
            return bytes(result)

    def close(self):
        if self.closed:
            return
        self.closed = True
        code = 0
        if self.process is not None:
            self.process.stdin.close()
            code = self.process.wait()
        self.log.close()
        self.events.close()
        if code != 0:
            raise RuntimeError(f"BEMU exited with status {code}: {self.log_dir}")


class Pool:
    def __init__(self, simulator: Path, artifact: dict, log_root: Path,
                 tile_indices: list[int], record_io: bool, arguments: list[tuple[str, ...]], memory_mib: int = 3072,
                 p2e: dict | None = None):
        if not tile_indices or len(set(tile_indices)) != len(tile_indices) or min(tile_indices) < 0:
            raise ValueError("tile indices must be nonempty, unique and nonnegative")
        if len(arguments) != len(tile_indices):
            raise ValueError("each tile requires its own program arguments")
        self.transport = None
        if p2e is not None:
            from .transport import Transport
            if tile_indices != list(range(len(tile_indices))):
                raise ValueError("P2E workers require controllers in rank order starting at zero")
            self.transport = Transport(Path(p2e["console_socket"]), len(tile_indices), p2e["timeout"])
        self.executors = [Executor(simulator, Path(artifact["program"]), Path(artifact["directory"]),
                                   log_root, tile, record_io, argument, memory_mib, self.transport, rank)
                          for rank, (tile, argument) in enumerate(zip(tile_indices, arguments))]
        self.threads = ThreadPoolExecutor(max_workers=len(tile_indices), thread_name_prefix="tile")
        self.next_tile = 0
        atexit.register(self.close)

    def submit(self, request: bytes, result_bytes: int):
        executor = self.executors[self.next_tile]
        self.next_tile = (self.next_tile + 1) % len(self.executors)
        return self.threads.submit(executor.execute, request, result_bytes)

    def submit_to(self, rank: int, request: bytes, result_bytes: int):
        return self.threads.submit(self.executors[rank].execute, request, result_bytes)

    def close(self):
        self.threads.shutdown(wait=True)
        for executor in self.executors:
            executor.close()
        if self.transport is not None:
            self.transport.close()


def run_program(package, args, arguments):
    log_root = args.log_dir.resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    if hasattr(args, "tile_indices"):
        if len(args.tile_indices) != 1:
            raise ValueError("native model execution requires exactly one tile index")
        selector, index, scope = "--tile-index", args.tile_indices[0], "tile"
        working_directory = []
    else:
        selector, index, scope = "--core-index", args.core_index, "core"
        working_directory = None
    directory = Path(tempfile.mkdtemp(prefix=f"{scope}-{index}-", dir=log_root))
    if working_directory is None:
        working_directory = ["--working-directory", str(directory)]
    for resource in package.directory.rglob("*"):
        target = directory / resource.relative_to(package.directory)
        if resource.is_dir():
            target.mkdir(parents=True)
        else:
            target.symlink_to(resource.resolve())
    for kind in ("cycle", "tensor"):
        (directory / "trace" / kind).mkdir(parents=True)
    program = directory / package.execution["emulation_program"] if "emulation_program" in package.execution else package.program
    with (directory / "simulator.log").open("wb") as log:
        subprocess.run([str(args.simulator.resolve()), "--elf", str(program),
                        "--log-dir", str(directory), *working_directory,
                        selector, str(index),
                        "--", *arguments], stderr=log, cwd=directory, check=True, timeout=args.timeout)
