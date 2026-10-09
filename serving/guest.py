"""Owned Linux BEMU execution for model packages using the Ant task runtime."""

import filecmp
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time


def firmware_profile(repo, chip, model, firmware=None, memory_mib=None):
    topology = json.loads(
        (
            Path(repo)
            / "examples/chips"
            / chip
            / "configs/generated/config/derived.json"
        ).read_text()
    )
    expected_harts = sum(hart["visible"] for hart in topology["harts"])
    matches = []
    incomplete = []
    requested = Path(firmware).resolve() if firmware is not None else None
    for cache in (Path(repo) / "bb-tests/build").glob("kernel*/CMakeCache.txt"):
        if requested is not None:
            rules = cache.parent / "CMakeFiles/kernel-build.dir/build.make"
            if (
                not rules.is_file()
                or f"{requested.with_suffix('.bin')}:" not in rules.read_text()
            ):
                continue
        values = {}
        for line in cache.read_text().splitlines():
            if line.startswith(("#", "//")) or "=" not in line or ":" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
        if requested is not None:
            required = {
                "BUCKYBALL_KERNEL_CHIP",
                "BUCKYBALL_KERNEL_MODEL",
                "BUCKYBALL_MODEL_STORAGE",
                "BUCKYBALL_KERNEL_INTERACTIVE",
                "BUCKYBALL_GUEST_MEMORY_MIB",
                "BUCKYBALL_HART_COUNT",
            }
            if required - values.keys():
                raise ValueError(
                    f"Incomplete matching kernel profile {cache}: missing {sorted(required - values.keys())}"
                )
        if (
            values.get("BUCKYBALL_KERNEL_CHIP") != chip
            or (model is not None and values.get("BUCKYBALL_KERNEL_MODEL") != model)
            or values.get("BUCKYBALL_MODEL_STORAGE") not in {"initramfs", "ddr"}
            or values.get("BUCKYBALL_KERNEL_INTERACTIVE") != "OFF"
        ):
            continue
        required = {
            "BUCKYBALL_GUEST_MEMORY_MIB",
            "BUCKYBALL_HART_COUNT",
            "BUCKYBALL_KERNEL_MODEL",
        }
        missing = required - values.keys()
        if missing:
            message = f"Incomplete kernel profile {cache}: missing {sorted(missing)}"
            if requested is not None:
                raise ValueError(message)
            incomplete.append(message)
            continue
        if not values["BUCKYBALL_KERNEL_MODEL"]:
            continue
        memory = int(values["BUCKYBALL_GUEST_MEMORY_MIB"])
        count = int(values["BUCKYBALL_HART_COUNT"])
        if count != expected_harts:
            continue
        name = (
            "fw_payload"
            + (f"-h{count}" if count != 64 else "")
            + "-"
            + values["BUCKYBALL_KERNEL_MODEL"]
        )
        if memory != 512:
            name += f"-mem{memory}M"
        if values["BUCKYBALL_MODEL_STORAGE"] == "ddr":
            name += "-ddr"
        image = Path(repo) / "bb-tests/output/kernel" / chip / f"{name}.elf"
        if firmware is not None and image.resolve() != Path(firmware).resolve():
            continue
        if memory_mib is not None and memory != memory_mib:
            continue
        if image.is_file() and (cache.parent / "rootfs/root/layout.json").is_file():
            matches.append((image, memory, cache.parent))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one built Linux profile for {chip}/{model}; found {len(matches)}. "
            "Specify --firmware and --guest-memory-mib or build the exact model kernel. "
            + "; ".join(incomplete)
        )
    return matches[0]


def validate_firmware(image, profile, package, load_manifest=None):
    rootfs = profile / "rootfs/root"
    layout = json.loads((rootfs / "layout.json").read_text())
    if layout["chip"] != package.chip or layout["execution"] != package.execution:
        raise ValueError(
            "Linux firmware model execution metadata differs from the selected package"
        )
    programs = list(rootfs.glob("*-run"))
    if len(programs) != 1 or not filecmp.cmp(
        programs[0], package.program, shallow=False
    ):
        raise ValueError("Linux firmware model ELF differs from the selected package")
    ddr = image.name.endswith("-ddr.elf")
    if load_manifest is not None and not ddr:
        raise ValueError("A DDR load manifest requires a DDR kernel profile")
    manifest = None
    if ddr:
        manifest = image.with_suffix(".load.json")
        if (
            load_manifest is not None
            and Path(load_manifest).resolve() != manifest.resolve()
        ):
            raise ValueError("Load manifest differs from the selected firmware profile")
        contents = json.loads(manifest.read_text())
        loads = {entry["role"]: entry for entry in contents["loads"]}
        if set(loads) != {"boot", "model"} or len(contents["loads"]) != 2:
            raise ValueError("DDR firmware requires exactly boot/model loads")
        for entry in loads.values():
            path = Path(entry["file"])
            if not path.is_absolute():
                path = manifest.parent / path
            if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"DDR load bytes differ from manifest: {path}")
        boot = Path(loads["boot"]["file"])
        model = Path(loads["model"]["file"])
        if not boot.is_absolute():
            boot = manifest.parent / boot
        if not model.is_absolute():
            model = manifest.parent / model
        if boot.resolve() != image.with_suffix(".bin").resolve():
            raise ValueError("DDR boot binary is not the selected firmware")
        index = json.loads(model.with_suffix(".files.json").read_text())
        if (
            index["model_base"] != contents["model_base"]
            or index["size"] != loads["model"]["size"]
        ):
            raise ValueError("DDR resource index geometry differs from load manifest")
        rows = [
            f'BBMODEL1\t{index["model_base"]}\t{index["size"]}\t{index["image_sha256"]}\n'
        ]
        rows.extend(
            f'{f["path"]}\t{f["offset"]}\t{f["size"]}\t{f["sha256"]}\n'
            for f in index["files"]
        )
        if model.with_suffix(".index").read_text() != "".join(rows) or not filecmp.cmp(
            model.with_suffix(".index"), rootfs / "model.index", shallow=False
        ):
            raise ValueError("Installed DDR resource index differs from packed model")
        if index["image_sha256"] != loads["model"]["sha256"]:
            raise ValueError("DDR resource index belongs to a different DDR image")
        indexed = {record["path"]: record for record in index["files"]}
        if len(indexed) != len(index["files"]) or set(indexed) != set(
            layout["resources"].values()
        ):
            raise ValueError("DDR resource index differs from package layout")
        for relative, record in indexed.items():
            path = checked_resource(package.directory, relative)
            if (
                path.stat().st_size != record["size"]
                or sha256(path) != record["sha256"]
            ):
                raise ValueError(
                    f"DDR resource differs from selected package: {relative}"
                )
        if not filecmp.cmp(
            profile / "guest/resource-check",
            rootfs / "guest/resource-check",
            shallow=False,
        ):
            raise ValueError(
                "Installed resource verifier differs from its built executable"
            )
    else:
        for relative in layout["resources"].values():
            if not filecmp.cmp(
                checked_resource(rootfs, relative),
                checked_resource(package.directory, relative),
                shallow=False,
            ):
                raise ValueError(f"Linux firmware model resource differs: {relative}")
    worker = layout["execution"].get("p2e", {}).get("kind") == "host-worker"
    if worker:
        mux = profile / "guest/worker-mux"
        if not filecmp.cmp(mux, rootfs / "guest/worker-mux", shallow=False):
            raise ValueError("Installed guest mux differs from its built executable")
        repo = Path(__file__).resolve().parents[2]
        if (
            mux.stat().st_mtime_ns
            < (repo / "stack/serving/loader/guest/mux.cpp").stat().st_mtime_ns
        ):
            raise ValueError("Guest mux was built before its current source")
    built = profile / "opensbi/platform/buckyball/firmware/fw_payload.elf"
    installed_files = [rootfs / "layout.json", rootfs / "run-model", *programs]
    if worker:
        installed_files.append(rootfs / "guest/worker-mux")
    for installed in installed_files:
        if built.stat().st_mtime_ns < installed.stat().st_mtime_ns:
            raise ValueError(f"Firmware predates an installed model input: {installed}")
    if not filecmp.cmp(image, built, shallow=False):
        raise ValueError("Selected firmware ELF differs from its kernel build profile")
    if ddr and not filecmp.cmp(
        boot,
        profile / "opensbi/platform/buckyball/firmware/fw_payload.bin",
        shallow=False,
    ):
        raise ValueError("DDR boot binary differs from its kernel build profile")
    return manifest


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_resource(directory, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid firmware model resource path")
    return directory / path


class LinuxGuest:
    def __init__(
        self,
        simulator,
        firmware,
        memory_mib,
        log_dir,
        timeout,
        itrace=False,
        mtrace=False,
        load_manifest=None,
        *,
        native=False,
        output=None,
        result_count=None,
    ):
        self.native = native
        self.output = output
        self.result_count = result_count
        self.log_dir = Path(log_dir)
        if timeout <= 0:
            raise ValueError("Linux guest timeout must be positive")
        self.timeout = timeout
        if load_manifest is not None:
            memory_mib = json.loads(Path(load_manifest).read_text())["ddr_size"] // (
                1024 * 1024
            )
        self.memory_mib = memory_mib
        self.command = [
            str(simulator),
            "--elf",
            str(firmware),
            "--log-dir",
            str(self.log_dir),
            *(["--itrace"] if itrace else []),
            *(["--mtrace"] if mtrace else []),
        ]
        if load_manifest is not None:
            self.command.extend(["--load-manifest", str(load_manifest)])
        else:
            self.command.extend(["--memory-mib", str(memory_mib)])
        self.process = None
        self.log = None

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        marker = self.log_dir / "console.sock.path"
        if marker.exists():
            raise ValueError(
                f"Linux model log directory contains an old console marker: {marker}"
            )
        self.log = (self.log_dir / "linux.log").open("xb")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if self.native:
                return self
            deadline = time.monotonic() + self.timeout
            cursors = {
                self.log_dir / "linux.log": 0,
                self.log_dir / "uart/hart-0.log": 0,
            }
            tails = {path: b"" for path in cursors}
            listening = False
            while not marker.exists() or not listening:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Linux BEMU exited before console startup ({self.process.returncode}): {self.log_dir}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Linux BEMU did not reach BBMUX1 LISTENING before timeout: {self.log_dir}"
                    )
                for path, offset in cursors.items():
                    if not path.exists():
                        continue
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        chunk = stream.read(65536)
                        cursors[path] = stream.tell()
                    data = tails[path] + chunk
                    listening |= b"BBMUX1 LISTENING\n" in data
                    tails[path] = data[-32:]
                time.sleep(0.05)
            socket = Path(marker.read_text().strip())
            if not stat.S_ISSOCK(socket.stat().st_mode):
                raise RuntimeError(
                    f"Linux BEMU console marker is not a socket: {socket}"
                )
            return {"console_socket": str(socket), "timeout": self.timeout}
        except BaseException:
            self.stop()
            self.log.close()
            raise

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.process.wait()
            return
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def __exit__(self, kind, value, traceback):
        try:
            if kind is not None:
                self.stop()
            else:
                try:
                    code = self.process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired as error:
                    self.stop()
                    raise TimeoutError(
                        f"Linux guest did not power down after model completion: {self.log_dir}"
                    ) from error
                if code:
                    raise RuntimeError(
                        f"Linux guest exited with status {code}: {self.log_dir}"
                    )
                if self.native and self.output == "json":
                    uart = self.log_dir / "uart/hart-0.log"
                    records = [
                        json.loads(line)
                        for line in uart.read_text().splitlines()
                        if line.startswith("{")
                    ]
                    if not records:
                        raise ValueError("native guest produced no JSON results")
                    if len(records) != self.result_count:
                        raise ValueError(
                            f"native guest produced {len(records)} results; expected {self.result_count}"
                        )
                    with (self.log_dir / "model-results.jsonl").open("w") as results:
                        for record in records:
                            line = json.dumps(record)
                            results.write(line + "\n")
                            print(line)

        finally:
            self.log.close()
