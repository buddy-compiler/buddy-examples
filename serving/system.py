import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class Connection:
    def __init__(self, directory: Path, chip: int, tile: int, timeout: float):
        path = directory / f"chip-{chip}" / f"tile-{tile}" / "io.sock"
        deadline = time.monotonic() + timeout
        while not path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Tile endpoint did not start: {path}")
            time.sleep(0.01)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        # Bind/connect through a directory fd to avoid Unix socket path limits.
        import os
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            self.socket.connect(f"/proc/self/fd/{descriptor}/{path.name}")
        finally:
            os.close(descriptor)
        self.lock = threading.Lock()

    def execute(self, request: bytes, result_bytes: int) -> bytes:
        with self.lock:
            self.socket.sendall(request)
            result = bytearray()
            while len(result) < result_bytes:
                chunk = self.socket.recv(result_bytes - len(result))
                if not chunk:
                    raise RuntimeError("Tile closed before completing its response")
                result.extend(chunk)
            return bytes(result)

    def close(self):
        with self.lock:
            self.socket.close()


class System:
    def __init__(self, simulator: Path, program: Path, model_dir: Path,
                 log_root: Path, memory_mib: int, timeout: float):
        log_root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="system-", dir=log_root))
        self.log = (self.directory / "simulator.log").open("wb")
        self.timeout = timeout
        self.process = subprocess.Popen(
            [str(simulator), "--elf", str(program), "--pk", "--host-io",
             "--memory-mib", str(memory_mib), "--log-dir", str(self.directory),
             "--", str(model_dir)],
            stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log,
        )

    def close(self, *, abort=False):
        try:
            if abort:
                self.process.terminate()
            try:
                code = self.process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise
        finally:
            self.log.close()
        if code != 0 and not abort:
            raise RuntimeError(f"BEMU system exited with status {code}: {self.directory}")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(abort=kind is not None)
