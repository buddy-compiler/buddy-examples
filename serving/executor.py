import atexit
import json
import math
import os
import select
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path


class Executor:
    def __init__(self, simulator: Path, program: Path, model_dir: Path, log_root: Path,
                 tile_index: int = 0, record_io: bool = False, arguments: tuple[str, ...] = (), memory_mib: int = 3072,
                 transport=None, rank: int = 0, itrace: bool = False, mtrace: bool = False,
                 *, timeout: float = 7200):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request timeout must be finite and positive")
        self.timeout = timeout
        self.failed = False
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
             *(["--itrace"] if itrace else []), *(["--mtrace"] if mtrace else []),
             "--", str(model_dir), *arguments],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            bufsize=0,
        )
        if self.process is not None:
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
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
                deadline = time.monotonic() + self.timeout
                try:
                    pending = memoryview(request)
                    while pending:
                        self._ready(self.process.stdin.fileno(), deadline, write=True)
                        try:
                            written = os.write(self.process.stdin.fileno(), pending[:65536])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError as error:
                            raise RuntimeError(f"BEMU input closed: {self.log_dir}") from error
                        if not written:
                            raise RuntimeError(f"BEMU input closed: {self.log_dir}")
                        pending = pending[written:]
                    result = bytearray()
                    while len(result) < result_bytes:
                        self._ready(self.process.stdout.fileno(), deadline, write=False)
                        try:
                            chunk = os.read(self.process.stdout.fileno(), min(65536, result_bytes - len(result)))
                        except BlockingIOError:
                            continue
                        if not chunk:
                            raise RuntimeError(f"BEMU exited before completing the request: {self.log_dir}")
                        result.extend(chunk)
                except BaseException:
                    self.failed = True
                    self._stop()
                    raise
            if self.record_io:
                (self.log_dir / f"{self.sequence:04d}-output.bin").write_bytes(result)
            self.events.write(json.dumps({"tile": self.tile_index, "sequence": self.sequence,
                                          "start_ns": started, "end_ns": time.monotonic_ns(),
                                          "request_bytes": len(request), "result_bytes": result_bytes}) + "\n")
            self.events.flush()
            self.sequence += 1
            return bytes(result)

    def _ready(self, descriptor, deadline, *, write):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"BEMU request timed out: {self.log_dir}")
        readers, writers, _ = select.select([] if write else [descriptor], [descriptor] if write else [], [], remaining)
        if not (readers or writers):
            raise TimeoutError(f"BEMU request timed out: {self.log_dir}")

    def _stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def close(self):
        if self.closed:
            return
        self.closed = True
        atexit.unregister(self.close)
        code = 0
        try:
            if self.process is not None:
                self.process.stdin.close()
                try:
                    code = self.process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    self.failed = True
                    self._stop()
                    raise TimeoutError(f"BEMU did not stop after input closed: {self.log_dir}") from None
                finally:
                    self.process.stdout.close()
        finally:
            self.log.close()
            self.events.close()
        if code != 0 and not self.failed:
            raise RuntimeError(f"BEMU exited with status {code}: {self.log_dir}")


class Pool:
    def __init__(self, simulator: Path, artifact: dict, log_root: Path,
                 tile_indices: list[int], record_io: bool, arguments: list[tuple[str, ...]], memory_mib: int = 3072,
                 p2e: dict | None = None, itrace: bool = False, mtrace: bool = False,
                 *, timeout: float = 7200):
        if not tile_indices or len(set(tile_indices)) != len(tile_indices) or min(tile_indices) < 0:
            raise ValueError("tile indices must be nonempty, unique and nonnegative")
        if len(arguments) != len(tile_indices):
            raise ValueError("each tile requires its own program arguments")
        self.transport = None
        if p2e is not None:
            from .transport import Transport
            if tile_indices != list(range(1, len(tile_indices) + 1)):
                raise ValueError("P2E workers require compute tiles in rank order after the main tile")
            self.transport = Transport(Path(p2e["console_socket"]), len(tile_indices), p2e["timeout"])
        self.executors = [Executor(simulator, Path(artifact["program"]), Path(artifact["directory"]),
                                   log_root, tile, record_io, argument, memory_mib, self.transport, rank,
                                   itrace, mtrace, timeout=timeout)
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
        atexit.unregister(self.close)
        if any(executor.failed for executor in self.executors):
            for executor in self.executors:
                executor._stop()
        self.threads.shutdown(wait=True, cancel_futures=True)
        with ExitStack() as cleanup:
            if self.transport is not None:
                cleanup.callback(self.transport.close)
            for executor in self.executors:
                cleanup.callback(executor.close)


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
                        *(["--itrace"] if args.itrace else []),
                        *(["--mtrace"] if args.mtrace else []),
                        "--", *arguments], stderr=log, cwd=directory, check=True, timeout=args.timeout)
