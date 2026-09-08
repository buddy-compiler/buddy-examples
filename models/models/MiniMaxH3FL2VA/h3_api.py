#!/usr/bin/env python3
"""Context-IR + Regenerate-2K helpers for MiniMax-H3 ModelTest runtime."""

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _req(method, url, body=None):
    token = os.environ.get("MINIMAX_API_KEY") or os.environ.get("TOKEN")
    if not token:
        raise SystemExit("MINIMAX_API_KEY (or TOKEN) is not set")
    base = os.environ.get("MINIMAX_API_BASE", "").rstrip("/")
    if not base:
        raise SystemExit("MINIMAX_API_BASE is not set")
    if url.startswith("/"):
        url = base + url
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {url}: {e.read().decode()}") from e


def _poll(task_id, timeout_s=1800):
    deadline = time.time() + timeout_s
    while True:
        result = _req("GET", f"/v2/query/video_generation/{task_id}")
        task = result.get("task")
        if not isinstance(task, dict):
            raise SystemExit(f"bad poll payload: {result}")
        status = task.get("status")
        if status == "succeeded":
            return result
        if status in ("failed", "cancelled", "error"):
            raise SystemExit(f"task {task_id} status={status}: {result}")
        if time.time() >= deadline:
            raise SystemExit(f"task {task_id} timed out")
        time.sleep(5)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ir = sub.add_parser("context-ir")
    ir.add_argument("--prompt", required=True)
    ir.add_argument("--duration", type=int, required=True)
    ir.add_argument("--ratio", required=True)
    ir.add_argument("--out", type=Path, required=True)
    ir.add_argument("--image", action="append", default=[])
    ir.add_argument("--image-role", action="append", default=[])

    r2k = sub.add_parser("regenerate-2k")
    r2k.add_argument("--prompt-file", type=Path, required=True)
    r2k.add_argument("--video", type=Path, required=True)
    r2k.add_argument("--out", type=Path, required=True)

    tok = sub.add_parser("tokenize")
    tok.add_argument("--model", required=True)
    tok.add_argument("--subfolder", default="")
    tok.add_argument("--prompt-file", type=Path, required=True)
    tok.add_argument("--ids-out", type=Path, required=True)
    tok.add_argument("--mask-out", type=Path, required=True)
    tok.add_argument("--max-len", type=int, required=True)

    args = p.parse_args()
    if args.cmd == "tokenize":
        import numpy
        from transformers import AutoProcessor

        if not args.prompt_file.is_file():
            raise SystemExit(f"prompt file not found: {args.prompt_file}")
        text = args.prompt_file.read_text(encoding="utf-8")
        if not text.strip():
            raise SystemExit("empty prompt file")
        model = args.model
        local = Path(model)
        kwargs = {"trust_remote_code": True}
        if local.exists():
            if args.subfolder and (local / args.subfolder).is_dir():
                local = local / args.subfolder
            processor = AutoProcessor.from_pretrained(str(local), **kwargs)
        else:
            if args.subfolder:
                kwargs["subfolder"] = args.subfolder
            processor = AutoProcessor.from_pretrained(model, **kwargs)
        out = processor(text=text, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=args.max_len)
        if "input_ids" not in out or "attention_mask" not in out:
            raise SystemExit(f"processor missing ids/mask: {list(out.keys())}")
        ids = out["input_ids"].cpu().numpy().astype(numpy.int64).reshape(-1)
        mask = out["attention_mask"].cpu().numpy().astype(numpy.int64).reshape(-1)
        if ids.size != args.max_len or mask.size != args.max_len:
            raise SystemExit(
                f"tokenize length mismatch: ids={ids.size} mask={mask.size} "
                f"max_len={args.max_len}"
            )
        ids.tofile(args.ids_out)
        mask.tofile(args.mask_out)
        print(args.ids_out)
        return

    if args.cmd == "context-ir":
        content = [{"type": "text", "text": args.prompt}]
        if len(args.image) != len(args.image_role):
            raise SystemExit("--image and --image-role count mismatch")
        for path, role in zip(args.image, args.image_role):
            img = Path(path)
            if not img.is_file():
                raise SystemExit(f"image not found: {img}")
            b64 = base64.b64encode(img.read_bytes()).decode()
            ext = img.suffix.lower().lstrip(".")
            mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                    "role": role,
                }
            )
        created = _req(
            "POST",
            "/v2/h3_context_ir",
            {
                "model": "MiniMax-H3",
                "content": content,
                "duration": args.duration,
                "ratio": args.ratio,
            },
        )
        task_id = created.get("task_id")
        if not task_id:
            raise SystemExit(f"context-ir missing task_id: {created}")
        result = _poll(str(task_id))
        prompt = result.get("task", {}).get("content", {}).get("prompt")
        if not prompt:
            raise SystemExit(f"context-ir missing prompt: {result}")
        args.out.write_text(prompt, encoding="utf-8")
        print(args.out)
        return

    if not args.prompt_file.is_file():
        raise SystemExit(f"prompt file not found: {args.prompt_file}")
    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise SystemExit("empty prompt file")
    b64 = base64.b64encode(args.video.read_bytes()).decode()
    created = _req(
        "POST",
        "/v2/video_regeneration",
        {
            "model": "MiniMax-H3",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                    "role": "base_video",
                },
            ],
            "resolution": "2K",
        },
    )
    task_id = created.get("task_id")
    if not task_id:
        raise SystemExit(f"regenerate-2k missing task_id: {created}")
    result = _poll(str(task_id))
    url = result.get("task", {}).get("content", {}).get("url")
    if not url:
        raise SystemExit(f"regenerate-2k missing url: {result}")
    with urllib.request.urlopen(url, timeout=600) as resp:
        args.out.write_bytes(resp.read())
    if args.out.stat().st_size == 0:
        raise SystemExit("downloaded empty 2K video")
    print(args.out)


if __name__ == "__main__":
    main()
