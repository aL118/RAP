"""Hand-draw 2D boxes for a clip the detector missed, as keyframed tracks.

GroundingDINO fails outright on some footage -- closetruck is 4 detections
across 60 frames, the truck backlit into blown-out white and half out of frame.
This serves a small annotator over the clip's frames so those objects can be
drawn by hand; apply_manual_boxes.py then turns what is drawn into masks and
merges them into mask_results_preds.json, upstream of the lift, so stages 3, 3b
and the navsim export need no knowledge that a box came from a person.

    python annotate_boxes.py --clip closetruck --dataset CARE_YTB --run 1

It binds 127.0.0.1 and prints a url. Under VS Code Remote the port is forwarded
automatically; over plain ssh, forward it yourself:

    ssh -N -L 8791:127.0.0.1:8791 <host>

Boxes are keyframed, not drawn per frame: a track carries a box on the frames
you place one on and is interpolated linearly across the gaps between them. A
clip like closetruck is then four or five drags rather than sixty, which is the
difference between the tool being used and not. Annotating every frame is still
available -- a keyframe on every frame is just a track with no gaps to fill.

A track can instead be marked a *reject region*, which deletes rather than
adds: every detector box it covers is dropped at merge time and it contributes
no box of its own. Drawing over a false positive cannot do that -- it swaps your
box in for theirs -- so a detection on the ego bonnet, or a hedge read as a car,
needs a region that says "nothing here is real" rather than a correction.
Reject regions are keyframed and interpolated like any other track.

Nothing here is destructive: the only file written is manual_boxes.json in the
run directory, the detector's own output is never touched, and the merge that
does touch it is a separate step you run when you are happy with the boxes.
"""

import argparse
import ast
import json
import os
import re
import socket
import webbrowser
from functools import partial

import box_schema
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PAGE = ROOT / "annotate_boxes.html"
PAGE_3D = ROOT / "annotate_boxes_3d.html"


def detection_categories():
    """infer_frames.DEFAULT_CATEGORIES, read out of the source rather than imported.

    The annotator must offer the same vocabulary the lift knows -- a category
    outside it lands a box with no size prior and no palette entry -- but
    importing infer_frames drags in torch and the whole GroundingDINO package
    for a list of sixteen strings: 29 seconds of startup, a page of unrelated
    warnings, and a hard dependency on being run from vis3d/ (grounding_sam.py
    builds its sys.path from os.getcwd()). Parsing the literal costs none of
    that and still fails loudly if the name is ever renamed.
    """
    tree = ast.parse((ROOT / "infer_frames.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DEFAULT_CATEGORIES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("error: DEFAULT_CATEGORIES not found in infer_frames.py")


DEFAULT_CATEGORIES = detection_categories()

FRAME_RE = re.compile(r"^[0-9A-Za-z_.-]+\.(jpg|jpeg|png)$", re.I)


def frame_list(frames_dir: Path):
    return sorted(p.name for p in frames_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def detector_boxes(output_dir: Path):
    """DINO's own boxes per frame, as xyxy, for display underneath the manual ones.

    Shown greyed out and not editable. The point is not to correct them one by
    one -- it is to see at a glance which frames the detector actually covered,
    so the frames worth drawing on are obvious.
    """
    preds_path = output_dir / "mask_results_preds.json"
    if not preds_path.exists():
        return {}
    with open(preds_path) as f:
        preds = json.load(f)
    out = {}
    for det in preds:
        if det.get("manual"):
            continue          # a previous merge's own boxes; not the detector's
        x, y, w, h = mask_utils.toBbox(det["mask"]).tolist()
        out.setdefault(det["frame"], []).append({
            "box": [x, y, x + w, y + h],
            "category": det["category"],
            "score": round(float(det["score"]), 3),
        })
    return out


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, ctx=None, **kw):
        self.ctx = ctx
        super().__init__(*a, **kw)

    def log_message(self, fmt, *args):      # one line per save, not per frame GET
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        ctx = self.ctx
        if path in ("/", "/index.html"):
            page = PAGE_3D if ctx["boxes3d"] else PAGE
            return self._send(200, page.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/meta3d":
            # boxes_3d.json wholesale: it carries the per-frame camera the page
            # needs to project a box, and a clip's worth is a few hundred KB --
            # smaller than one of the frames it is drawn over.
            boxes = json.loads((ctx["output_dir"] / "boxes_3d.json").read_text())
            boxes.pop("_manual_3d", None)
            # The page indexes a box positionally (box[0], box[1], ...), so it is
            # handed the array form rather than the file's named fields. What it
            # writes back to manual_boxes_3d.json is arrays too, which
            # apply_manual_boxes_3d.py reads through the same box_schema.
            for entry in boxes.values():
                if isinstance(entry, dict) and "boxes" in entry:
                    entry["boxes"] = [list(box_schema.from_any(b)) for b in entry["boxes"]]
            return self._send(200, json.dumps({
                "clip": ctx["clip"], "dataset": ctx["dataset"], "run": ctx["run"],
                "frames": [f for f in ctx["frames"] if f in boxes],
                "categories": DEFAULT_CATEGORIES,
                "boxes": boxes,
                "manual": ctx["manual3d_path"].exists()
                          and json.loads(ctx["manual3d_path"].read_text()) or None,
                "save_path": str(ctx["manual3d_path"]),
            }))

        if path == "/api/meta":
            return self._send(200, json.dumps({
                "clip": ctx["clip"],
                "dataset": ctx["dataset"],
                "run": ctx["run"],
                "frames": ctx["frames"],
                "categories": DEFAULT_CATEGORIES,
                "detector": detector_boxes(ctx["output_dir"]),
                "manual": ctx["manual_path"].exists()
                          and json.loads(ctx["manual_path"].read_text()) or None,
                "save_path": str(ctx["manual_path"]),
            }))

        if path.startswith("/frames/"):
            name = path[len("/frames/"):]
            # The frame name comes off the wire, so it decides a path on disk:
            # anything but a plain image basename is refused rather than
            # normalised, and the resolved path is checked to be inside
            # frames/ so a name that survives the pattern still cannot escape.
            if not FRAME_RE.match(name):
                return self._send(404, b"", "text/plain")
            frame_path = (ctx["frames_dir"] / name).resolve()
            if frame_path.parent != ctx["frames_dir"] or not frame_path.exists():
                return self._send(404, b"", "text/plain")
            ctype = "image/png" if name.lower().endswith(".png") else "image/jpeg"
            return self._send(200, frame_path.read_bytes(), ctype,
                              {"Cache-Control": "max-age=3600"})

        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/save3d":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError) as exc:
                return self._send(400, json.dumps({"error": f"bad json: {exc}"}))
            ok, err = validate_3d(payload, self.ctx["frames"])
            if not ok:
                return self._send(400, json.dumps({"error": err}))
            payload["clip"] = self.ctx["clip"]
            payload["run"] = self.ctx["run"]
            out = self.ctx["manual3d_path"]
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, out)
            n = sum(len(v) for v in payload["edits"].values())
            print(f"saved {n} edit(s) over {len(payload['edits'])} frame(s) -> {out}")
            return self._send(200, json.dumps({"edits": n, "frames": len(payload["edits"])}))
        if path != "/api/save":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError) as exc:
            return self._send(400, json.dumps({"error": f"bad json: {exc}"}))

        ok, err = validate(payload, self.ctx["frames"])
        if not ok:
            return self._send(400, json.dumps({"error": err}))

        payload["clip"] = self.ctx["clip"]
        out = self.ctx["manual_path"]
        # Written through a temp file in the same directory: a half-written
        # manual_boxes.json is the one file here that cannot be regenerated.
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, out)
        n_kf = sum(len(t["keyframes"]) for t in payload["tracks"])
        print(f"saved {len(payload['tracks'])} track(s), {n_kf} keyframe(s) -> {out}")
        return self._send(200, json.dumps({"saved": str(out),
                                           "tracks": len(payload["tracks"])}))


def validate(payload, frames):
    """Reject a payload the merge step could not act on, while it can still be fixed.

    The annotator is the only writer, so this is not adversarial input -- it is
    a guard against a stale browser tab left open across a re-extract at a
    different frame rate, where every keyframe would name a frame that no
    longer exists and the merge would silently produce nothing.
    """
    known = set(frames)
    if not isinstance(payload, dict) or not isinstance(payload.get("tracks"), list):
        return False, "payload needs a 'tracks' list"
    for i, track in enumerate(payload["tracks"]):
        if not isinstance(track, dict):
            return False, f"track {i}: not an object"
        if not isinstance(track.get("keyframes"), dict):
            return False, f"track {i}: no keyframes"
        # A reject region has no class: it names no object, it deletes whatever
        # the detector thought was there. Requiring one would mean picking a
        # category that then has to be ignored everywhere downstream.
        if not track.get("reject") and track.get("category") not in DEFAULT_CATEGORIES:
            return False, f"track {i}: unknown category {track.get('category')!r}"
        for frame, box in track["keyframes"].items():
            if frame not in known:
                return False, f"track {i}: no such frame {frame!r}"
            if box is None:
                continue                      # explicit "gone from here"
            if (not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(v, (int, float)) for v in box)):
                return False, f"track {i}, {frame}: box must be [x0,y0,x1,y1]"
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                return False, f"track {i}, {frame}: box is degenerate"
    return True, None


def validate_3d(payload, frames):
    """Reject a 3D edit payload the apply step could not act on.

    Same purpose as validate(): the annotator is the only writer, so this guards
    against a stale tab rather than against malice -- but a box with five
    numbers in it would be written straight into boxes_3d.json and only fail
    much later, inside the raster or the export.
    """
    known = set(frames)
    if not isinstance(payload, dict) or not isinstance(payload.get("edits"), dict):
        return False, "payload needs an 'edits' object"
    for frame, items in payload["edits"].items():
        if frame not in known:
            return False, f"no such frame {frame!r}"
        if not isinstance(items, list):
            return False, f"{frame}: edits must be a list"
        for i, edit in enumerate(items):
            op = edit.get("op")
            if op not in ("delete", "replace", "add"):
                return False, f"{frame}[{i}]: unknown op {op!r}"
            if op != "add":
                at = edit.get("at")
                if not (isinstance(at, list) and len(at) == 3
                        and all(isinstance(v, (int, float)) for v in at)):
                    return False, f"{frame}[{i}]: {op} needs an 'at' [x,y,z] anchor"
            if op != "delete":
                box = edit.get("box")
                if not (isinstance(box, list) and len(box) == 7
                        and all(isinstance(v, (int, float)) for v in box)):
                    return False, f"{frame}[{i}]: {op} needs a 7-number box"
                if min(box[3], box[4], box[5]) <= 0:
                    return False, f"{frame}[{i}]: box has a non-positive extent"
                if edit.get("name") not in DEFAULT_CATEGORIES:
                    return False, f"{frame}[{i}]: unknown category {edit.get('name')!r}"
    return True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="clip directory under data/<dataset>/")
    ap.add_argument("--dataset", default="CARE_YTB", help="dataset dir under data/")
    ap.add_argument("--run", default="1",
                    help="run subdir holding this clip's outputs ('' = the clip dir)")
    ap.add_argument("--boxes3d", action="store_true",
                    help="correct the lifted 3D boxes in this run's boxes_3d.json instead "
                         "of drawing 2D boxes for the detector. Saves to "
                         "manual_boxes_3d.json; apply with apply_manual_boxes_3d.py.")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--open", action="store_true",
                    help="also try to open a browser here (pointless over ssh)")
    args = ap.parse_args()

    clip_dir = BASE / "data" / args.dataset / args.clip
    frames_dir = clip_dir / "frames"
    output_dir = clip_dir / args.run if args.run else clip_dir
    if not frames_dir.is_dir():
        raise SystemExit(f"error: no frames at {frames_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = frame_list(frames_dir)
    if not frames:
        raise SystemExit(f"error: no frame images in {frames_dir}")

    ctx = {
        "clip": args.clip,
        "dataset": args.dataset,
        "run": args.run,
        "frames_dir": frames_dir.resolve(),
        "output_dir": output_dir,
        "frames": frames,
        "manual_path": output_dir / "manual_boxes.json",
        "manual3d_path": output_dir / "manual_boxes_3d.json",
        "boxes3d": args.boxes3d,
    }
    if args.boxes3d and not (output_dir / "boxes_3d.json").exists():
        raise SystemExit(f"error: no boxes_3d.json in {output_dir}; run stage 3 first")

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, ctx=ctx))
    except OSError as exc:
        # Overwhelmingly this is an annotator left running in another terminal
        # -- the tool is a long-lived server people forget to stop -- and a
        # socket traceback says none of that.
        raise SystemExit(f"error: cannot bind port {args.port}: {exc}\n"
                         f"       an annotator may already be running "
                         f"(pkill -f annotate_boxes.py), or use --port")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    if args.boxes3d:
        lifted = json.loads((output_dir / "boxes_3d.json").read_text())
        n = sum(len(v.get("boxes", [])) for k, v in lifted.items() if k != "_manual_3d")
        print(f"{args.clip}: {len(frames)} frames, {n} lifted 3D boxes")
    else:
        print(f"{args.clip}: {len(frames)} frames, "
              f"{sum(len(v) for v in detector_boxes(output_dir).values())} detector boxes")
    print(f"serving {url}   (ctrl-c to stop)")
    print(f"saves to {ctx['manual3d_path'] if args.boxes3d else ctx['manual_path']}")
    if socket.gethostname() and not args.open:
        print("if this is a remote host, forward the port: "
              f"ssh -N -L {args.port}:127.0.0.1:{args.port} {socket.gethostname()}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
