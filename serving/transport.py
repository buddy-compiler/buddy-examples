"""Framed worker RPC over the P2E console's binary UART stream."""
import socket
import struct
import threading
from concurrent.futures import Future, TimeoutError
from pathlib import Path

_HEADER = struct.Struct("<QQQQ")
_MAGIC = b"BBMUX1\n"
_MAX_FRAME = 256 * 1024 * 1024
_CLOSE = (1 << 64) - 1


class Transport:
    def __init__(self, console_socket: Path, workers: int, timeout: float):
        if workers < 1 or timeout <= 0:
            raise ValueError("transport requires workers and a positive timeout")
        self.workers = workers
        self.timeout = timeout
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.pending = {}
        self.sequence = [0] * workers
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.error = None
        self.closed = False
        self.finished = threading.Event()
        self.close_ack = False
        try:
            self.socket.connect(str(console_socket))
            self.socket.sendall(b"hart 0\n" + _MAGIC)
            if self._read(len(_MAGIC)) != _MAGIC:
                raise RuntimeError("P2E guest did not acknowledge the worker transport protocol")
        except BaseException:
            self.socket.close()
            raise
        self.socket.settimeout(None)
        self.reader = threading.Thread(target=self._receive, name="p2e-worker-responses", daemon=True)
        self.reader.start()

    def _read(self, size):
        result = bytearray()
        while len(result) < size:
            chunk = self.socket.recv(size - len(result))
            if not chunk:
                raise EOFError("P2E worker transport closed")
            result.extend(chunk)
        return bytes(result)

    def _fail(self, error):
        with self.lock:
            if self.error is None:
                self.error = error
            requests = list(self.pending.values())
            self.pending.clear()
        for _, future in requests:
            future.set_exception(error)
        self.finished.set()
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _receive(self):
        try:
            while True:
                rank, sequence, status, size = _HEADER.unpack(self._read(_HEADER.size))
                if rank == _CLOSE:
                    with self.lock:
                        if not self.closed or (sequence, status, size) != (0, 0, 0) or self.pending:
                            raise RuntimeError("invalid P2E close acknowledgement")
                        self.close_ack = True
                    self.finished.set()
                    return
                key = (rank, sequence)
                with self.lock:
                    if key not in self.pending:
                        raise RuntimeError(f"unexpected P2E response: rank={rank}, sequence={sequence}")
                    expected, future = self.pending[key]
                if (status == 0 and size != expected) or (status != 0 and size > 4096):
                    raise RuntimeError(f"invalid P2E response length: rank={rank}, status={status}, bytes={size}")
                payload = self._read(size)
                if status:
                    raise RuntimeError(f"P2E worker {rank} failed ({status}): {payload.decode('utf-8', errors='replace')}")
                with self.lock:
                    # A timeout/close may have failed all requests while recv was in flight.
                    if key not in self.pending:
                        return
                    del self.pending[key]
                future.set_result(payload)
        except Exception as error:
            self._fail(error)

    def execute(self, rank: int, request: bytes, result_bytes: int) -> bytes:
        if not 0 <= rank < self.workers or not 0 < result_bytes <= _MAX_FRAME or not 0 < len(request) <= _MAX_FRAME:
            raise ValueError("invalid worker rank or frame size (bbmux1 requires 1..256 MiB per frame)")
        future = Future()
        with self.lock:
            if self.closed or self.error:
                raise RuntimeError("P2E worker transport is unavailable") from self.error
            if any(key[0] == rank for key in self.pending):
                raise RuntimeError(f"worker {rank} already has an outstanding request")
            sequence = self.sequence[rank]
            self.sequence[rank] += 1
            self.pending[(rank, sequence)] = (result_bytes, future)
        # The timeout also covers a blocked UART send, not just response reception.
        timer = threading.Timer(self.timeout, self._fail,
                                args=(TimeoutError(f"P2E worker {rank} request {sequence} timed out"),))
        timer.start()
        try:
            with self.send_lock:
                self.socket.sendall(_HEADER.pack(rank, sequence, len(request), result_bytes))
                self.socket.sendall(request)
            return future.result()
        except Exception as error:
            self._fail(error)
            raise
        finally:
            timer.cancel()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            graceful = self.error is None and not self.pending
        timer = threading.Timer(self.timeout, self._fail,
                                args=(TimeoutError("P2E guest did not close its workers"),))
        try:
            if graceful:
                timer.start()
                with self.send_lock:
                    self.socket.sendall(_HEADER.pack(_CLOSE, 0, 0, 0))
                self.finished.wait()
                if not self.close_ack:
                    raise RuntimeError("P2E guest shutdown was not acknowledged") from self.error
        finally:
            timer.cancel()
            self._fail(RuntimeError("P2E worker transport closed by host"))
            self.socket.close()
            self.reader.join()
