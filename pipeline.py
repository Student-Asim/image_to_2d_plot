import io
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

import config as C

log = logging.getLogger("pipeline")
_horizonnet_model = None


def load_models():
    global _horizonnet_model
    sys.path.append(str(C.HORIZONNET_DIR))
    from model import HorizonNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HorizonNet(backbone="resnet50", use_rnn=True).to(device)
    ckpt = torch.load(C.HORIZONNET_WEIGHTS, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    model.eval()
    _horizonnet_model = model
    log.info("HorizonNet loaded on %s", device)


def run_horizonnet(img_pil: Image.Image):
    if _horizonnet_model is None:
        raise RuntimeError("HorizonNet not loaded.")

    device = next(_horizonnet_model.parameters()).device
    img_pil = img_pil.resize(C.PANORAMA_SIZE)
    img_np = np.array(img_pil)
    x = T.ToTensor()(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        y_bon, y_cor = _horizonnet_model(x)

    cor = torch.sigmoid(y_cor)[0, 0].cpu().numpy()
    bon = y_bon.cpu().numpy()[0]
    return cor, bon, img_np, img_pil


def _extract_stable_corners(cor_map):
    cor_map = cor_map.astype(np.float32)
    smooth = np.convolve(cor_map, np.ones(7) / 7, mode="same")
    peaks, _ = find_peaks(
        smooth,
        height=C.PEAK_THRESH,
        distance=C.MIN_PEAK_DISTANCE,
    )
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=np.float32), smooth

    clust = DBSCAN(eps=C.CLUSTER_EPS, min_samples=1).fit(peaks.reshape(-1, 1))
    stable = sorted(int(np.mean(peaks[clust.labels_ == lab])) for lab in np.unique(clust.labels_))
    stable = np.array(stable)
    return np.stack([stable, np.zeros_like(stable)], axis=1).astype(np.float32), smooth


def _order_polygon(pts):
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def _manhattanize(pts):
    pts = pts.copy()
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        d = p2 - p1
        if abs(d[0]) > abs(d[1]):
            p2[1] = p1[1]
        else:
            p2[0] = p1[0]
        pts[(i + 1) % len(pts)] = p2
    return pts


def _remove_small_edges(pts):
    cleaned = np.array([
        pts[i] for i in range(len(pts))
        if np.linalg.norm(pts[(i + 1) % len(pts)] - pts[i]) > C.MIN_EDGE_LEN
    ])
    return cleaned if len(cleaned) >= 3 else pts


def build_floorplan_polygon(cor, width):
    corners, smooth = _extract_stable_corners(cor)
    if len(corners) < 3:
        raise ValueError("Too few corners detected. Use a clearer panorama.")

    corner_pixels = corners[:, 0]
    corner_angles = (corner_pixels / width) * 2 * np.pi

    raw_pts = np.stack([np.cos(corner_angles), np.sin(corner_angles)], axis=1)
    raw_pts = _order_polygon(raw_pts)
    raw_pts = _manhattanize(raw_pts)
    raw_pts = _remove_small_edges(raw_pts)

    mins = raw_pts.min(axis=0)
    maxs = raw_pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    scale = C.ROOM_SCALE / span.max()
    pts_m = (raw_pts - mins) * scale

    return pts_m, len(pts_m), pts_m.mean(0), corner_pixels, smooth


def _map_rf_class(pred):
    cls_id = int(pred.get("class_id", -1))
    raw_class = str(pred.get("class", ""))
    return cls_id, C.RF_CLASS_MAP_ID.get(cls_id, f"class_{cls_id}"), raw_class


def _box_iou(a, b):
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    ub = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return inter / (ua + ub - inter + 1e-6)


def _nms(dets, iou_thresh=0.40, pano_w=None):
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep = []

    def _variants(d):
        if pano_w is None or not d.get("wraps", False):
            return [d]
        a, b, c = dict(d), dict(d), dict(d)
        a["x1"], a["x2"] = d["x1"] % pano_w, d["x2"] % pano_w + pano_w
        b["x1"], b["x2"] = d["x1"], d["x2"]
        c["x1"], c["x2"] = d["x1"] + pano_w, d["x2"] + pano_w
        return [a, b, c]

    def seam_iou(a, b):
        return max(_box_iou(va, vb) for va in _variants(a) for vb in _variants(b))

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if seam_iou(best, d) < iou_thresh]

    return keep


def _normalize_pano_box(global_cx, global_cy, local_w, local_h, pano_w, pano_h):
    half_w = local_w / 2.0
    half_h = local_h / 2.0
    x1 = global_cx - half_w
    x2 = global_cx + half_w
    y1 = max(0.0, global_cy - half_h)
    y2 = min(float(pano_h), global_cy + half_h)
    wraps = (x1 < 0.0) or (x2 > pano_w)
    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "wraps": wraps}


def detect_doors_windows(img_pil: Image.Image):
    if not C.ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is not set.")

    pano_w, pano_h = img_pil.size
    tile_step = C.TILE_W - C.OVERLAP
    tile_offsets = list(range(0, pano_w, tile_step))
    all_raw = []

    for ox in tile_offsets:
        if ox + C.TILE_W <= pano_w:
            tile_pil = img_pil.crop((ox, 0, ox + C.TILE_W, pano_h))
        else:
            right_part = img_pil.crop((ox, 0, pano_w, pano_h))
            left_w = C.TILE_W - (pano_w - ox)
            left_part = img_pil.crop((0, 0, left_w, pano_h))
            tile_pil = Image.new("RGB", (C.TILE_W, pano_h))
            tile_pil.paste(right_part, (0, 0))
            tile_pil.paste(left_part, (pano_w - ox, 0))

        buf = io.BytesIO()
        tile_pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        resp = requests.post(
            C.ROBOFLOW_URL,
            params={
                "api_key": C.ROBOFLOW_API_KEY,
                "confidence": C.CONF_THRESH,
                "overlap": C.OVERLAP_THR,
                "format": "json",
            },
            files={"file": ("tile.jpg", buf, "image/jpeg")},
            timeout=(5, 40),
        )
        resp.raise_for_status()
        preds = resp.json().get("predictions", [])

        for p in preds:
            cls_id, cls_name, raw_class = _map_rf_class(p)
            if cls_name not in {"door", "window"}:
                continue

            local_cx = float(p["x"])
            local_cy = float(p["y"])
            local_w = float(p["width"])
            local_h = float(p["height"])
            conf = float(p["confidence"])

            global_cx = (ox + local_cx) % pano_w
            box = _normalize_pano_box(global_cx, local_cy, local_w, local_h, pano_w, pano_h)

            all_raw.append({
                "class": cls_name,
                "class_id": cls_id,
                "raw_class": raw_class,
                "conf": conf,
                "x1": float(box["x1"]),
                "y1": float(box["y1"]),
                "x2": float(box["x2"]),
                "y2": float(box["y2"]),
                "cx": float(global_cx),
                "cy": float(local_cy),
                "width": float(local_w),
                "height": float(local_h),
                "wraps": bool(box["wraps"]),
            })

    all_raw = [d for d in all_raw if d["conf"] >= C.CONF_THRESH]

    final = []
    for cls_name in sorted(set(d["class"] for d in all_raw)):
        cls_dets = [d for d in all_raw if d["class"] == cls_name]
        final.extend(_nms(cls_dets, C.NMS_IOU, pano_w=pano_w))

    for d in final:
        d["x"] = d["cx"]
        d["y"] = d["cy"]
        d["confidence"] = d["conf"]

    return final


def _pano_x_to_angle(px, img_w=1024):
    return (px / img_w) * 2 * np.pi


def _ray_wall_intersect(origin, direction, p1, p2):
    wall = p2 - p1
    denom = direction[0] * wall[1] - direction[1] * wall[0]
    if abs(denom) < 1e-9:
        return None

    diff = p1 - origin
    t = (diff[0] * wall[1] - diff[1] * wall[0]) / denom
    u = (diff[0] * direction[1] - diff[1] * direction[0]) / denom

    if t > 1e-6 and 0 <= u <= 1:
        return t, u
    return None


def _find_hit_wall(origin, angle, pts_m, n_walls):
    d_vec = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    best = None
    for i in range(n_walls):
        res = _ray_wall_intersect(origin, d_vec, pts_m[i], pts_m[(i + 1) % n_walls])
        if res is None:
            continue
        t, u = res
        if best is None or t < best[0]:
            best = (t, u, i)
    return best


def _circular_delta(a, b):
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi


def _extract_layout_opening_candidates(corner_pixels, width, pts_m, n_walls, room_center):
    angles = np.sort((corner_pixels / width) * 2 * np.pi)
    if len(angles) < 2:
        return []

    boundaries = np.r_[angles, angles[0] + 2 * np.pi]
    candidates = []

    for a1, a2 in zip(boundaries[:-1], boundaries[1:]):
        mid = ((a1 + a2) / 2.0) % (2 * np.pi)
        hit = _find_hit_wall(room_center, mid, pts_m, n_walls)
        if hit is None:
            continue
        _, frac, wall_i = hit
        candidates.append({
            "angle": mid,
            "wall_i": wall_i,
            "frac": frac,
        })

    return candidates


def map_openings(dw_dets, pts_m, n_walls, room_center, width, height, corner_pixels):
    layout_candidates = _extract_layout_opening_candidates(
        corner_pixels,
        width,
        pts_m,
        n_walls,
        room_center,
    )

    mapped = []
    counts = defaultdict(int)
    neighborhood = np.radians(C.ALIGN_NEIGHBORHOOD_DEG)

    for det in sorted(dw_dets, key=lambda d: (-d["confidence"], d["class"])):
        cls = det["class"].lower()
        if cls not in {"door", "window"}:
            continue

        angle = _pano_x_to_angle(det["x"], width)
        direct_hit = _find_hit_wall(room_center, angle, pts_m, n_walls)
        if direct_hit is None:
            continue

        _, direct_frac, direct_wall = direct_hit

        best_score = 1e9
        best_wall = direct_wall
        best_frac = direct_frac

        for cand in layout_candidates:
            delta = abs(_circular_delta(angle, cand["angle"]))
            if delta > neighborhood:
                continue

            same_wall_bonus = 0.0 if cand["wall_i"] == direct_wall else 0.12
            vertical_penalty = (
                1.0 - min(max(det["cy"] / float(height), 0.0), 1.0)
            ) * C.ALIGN_VERTICAL_WEIGHT
            score = (
                delta
                + same_wall_bonus
                + vertical_penalty
                + abs(cand["frac"] - direct_frac) * C.ALIGN_CENTER_PULL
            )

            if score < best_score:
                best_score = score
                best_wall = cand["wall_i"]
                best_frac = cand["frac"]

        p1 = pts_m[best_wall]
        p2 = pts_m[(best_wall + 1) % n_walls]
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1e-6:
            continue

        margin = min(C.OPENING_MIN_WALL_MARGIN, wall_len * 0.2)
        frac_margin = margin / wall_len
        best_frac = float(np.clip(best_frac, frac_margin, 1.0 - frac_margin))
        center_pt = p1 + best_frac * wall_vec
        wall_dir = wall_vec / wall_len
        normal = np.array([-wall_dir[1], wall_dir[0]])

        counts[cls] += 1
        width_m = C.DOOR_WIDTH_M if cls == "door" else C.WINDOW_WIDTH_M
        width_m = min(width_m, max(0.25, wall_len * 0.7))
        tag = f"{'D' if cls == 'door' else 'W'}{counts[cls]}"

        mapped.append({
            "type": cls,
            "tag": tag,
            "conf": det["confidence"],
            "center": center_pt,
            "wall_i": best_wall,
            "frac": best_frac,
            "wall_dir": wall_dir,
            "normal": normal,
            "width": width_m,
            "wall_len": wall_len,
            "p1": p1,
            "p2": p2,
            "raw_class": det.get("raw_class"),
            "class_id": det.get("class_id"),
            "angle": angle,
            "pano_xy": [float(det["x"]), float(det["y"])],
            "direct_wall": direct_wall,
            "direct_frac": float(direct_frac),
        })

    return mapped


def save_layout_debug(pts_m, room_center, corner_pixels, img_w, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(plt.Polygon(pts_m, closed=True, fill=False, edgecolor="black", linewidth=2.0))
    ax.scatter(pts_m[:, 0], pts_m[:, 1], s=40, c="red", zorder=3)
    ax.scatter(room_center[0], room_center[1], s=50, c="blue", zorder=3)

    for idx, px in enumerate(corner_pixels):
        ang = _pano_x_to_angle(px, img_w)
        ray = room_center + np.array([np.cos(ang), np.sin(ang)]) * 1.4
        ax.plot([room_center[0], ray[0]], [room_center[1], ray[1]], "--", color="#b0b0b0", lw=1)
        ax.text(ray[0], ray[1], f"C{idx + 1}", fontsize=8)

    ax.set_aspect("equal")
    ax.set_title("HorizonNet polygon layout")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_openings_debug_overlay(img_np, mapped_openings, out_path: Path):
    vis = img_np.copy()
    h, w = vis.shape[:2]
    colors = {"door": (40, 80, 230), "window": (30, 170, 220)}

    for op in mapped_openings:
        x = int(np.clip(op["pano_xy"][0], 0, w - 1))
        y = int(np.clip(op["pano_xy"][1], 0, h - 1))
        color = colors.get(op["type"], (0, 255, 255))
        cv2.circle(vis, (x, y), 7, color, -1)
        cv2.putText(
            vis,
            f"{op['tag']} -> wall {op['wall_i'] + 1} {op['frac']:.2f}",
            (max(8, x + 8), max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), vis)


def _draw_wall_segment(ax, p1, p2, openings):
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return

    wall_dir = vec / length
    normal = np.array([-wall_dir[1], wall_dir[0]])
    thickness = C.WALL_THICKNESS

    cuts = sorted(
        (
            np.clip(op["frac"] - op["width"] / 2 / length, 0, 1),
            np.clip(op["frac"] + op["width"] / 2 / length, 0, 1),
        )
        for op in openings
    )

    segments, prev = [], 0.0
    for cut_start, cut_end in cuts:
        if cut_start > prev:
            segments.append((prev, cut_start))
        prev = max(prev, cut_end)
    if prev < 1.0:
        segments.append((prev, 1.0))

    for start, end in segments:
        if end - start < 1e-3:
            continue
        sp = p1 + start * vec
        ep = p1 + end * vec
        poly = np.array([
            sp + normal * thickness / 2,
            ep + normal * thickness / 2,
            ep - normal * thickness / 2,
            sp - normal * thickness / 2,
        ])
        ax.add_patch(plt.Polygon(poly, closed=True, fc="#111111", ec="#111111", lw=0.3, zorder=3))


def _draw_door(ax, op):
    p1 = op["p1"]
    wall_len = op["wall_len"]
    wall_dir = op["wall_dir"]
    normal = op["normal"]

    cut_start = np.clip(op["frac"] - op["width"] / 2 / wall_len, 0, 1)
    cut_end = np.clip(op["frac"] + op["width"] / 2 / wall_len, 0, 1)

    open_start = p1 + cut_start * (op["p2"] - op["p1"])
    open_end = p1 + cut_end * (op["p2"] - op["p1"])
    open_width = np.linalg.norm(open_end - open_start)

    leaf_end = np.array([
        open_start[0] + normal[0] * open_width,
        open_start[1] + normal[1] * open_width,
    ])
    ax.plot(
        [open_start[0], leaf_end[0]],
        [open_start[1], leaf_end[1]],
        color="#777777",
        lw=1.0,
        zorder=5,
    )

    theta = np.linspace(0, np.pi / 2, 50)
    base = np.arctan2(wall_dir[1], wall_dir[0])
    arc = np.c_[
        open_start[0] + np.cos(base + theta) * open_width,
        open_start[1] + np.sin(base + theta) * open_width,
    ]
    ax.plot(arc[:, 0], arc[:, 1], color="#777777", lw=0.9, zorder=5)


def _draw_window(ax, op):
    p1 = op["p1"]
    wall_len = op["wall_len"]
    normal = op["normal"]

    cut_start = np.clip(op["frac"] - op["width"] / 2 / wall_len, 0, 1)
    cut_end = np.clip(op["frac"] + op["width"] / 2 / wall_len, 0, 1)

    open_start = p1 + cut_start * (op["p2"] - op["p1"])
    open_end = p1 + cut_end * (op["p2"] - op["p1"])
    thickness = C.WALL_THICKNESS

    for off, lw in zip([-thickness / 2, 0, thickness / 2], [0.9, 1.2, 0.9]):
        wx1 = open_start + normal * off
        wx2 = open_end + normal * off
        ax.plot([wx1[0], wx2[0]], [wx1[1], wx2[1]], color="#666666", lw=lw, zorder=5)


def render_floorplan(pts_m, n_walls, mapped_openings, out_png: Path, out_dxf: Path):
    pad = 0.85
    xlim = (pts_m[:, 0].min() - pad, pts_m[:, 0].max() + pad)
    ylim = (pts_m[:, 1].min() - pad, pts_m[:, 1].max() + pad)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    ax.add_patch(plt.Polygon(pts_m, closed=True, fc="#fffdf9", ec="none", zorder=1))
    ax.grid(True, ls="-", alpha=0.45, color="#eee8de", lw=0.35)

    openings_by_wall = defaultdict(list)
    for op in mapped_openings:
        openings_by_wall[op["wall_i"]].append(op)

    for i in range(n_walls):
        _draw_wall_segment(ax, pts_m[i], pts_m[(i + 1) % n_walls], openings_by_wall[i])

    for op in mapped_openings:
        if op["type"] == "door":
            _draw_door(ax, op)
        else:
            _draw_window(ax, op)

    for i in range(n_walls):
        p1, p2 = pts_m[i], pts_m[(i + 1) % n_walls]
        mid = (p1 + p2) / 2
        ax.text(mid[0], mid[1], f"{np.linalg.norm(p2 - p1):.2f} m", fontsize=8, color="#666")

    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(str(out_png), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _export_dxf(pts_m, n_walls, mapped_openings, out_dxf)


def _export_dxf(pts_m, n_walls, mapped_openings, out_dxf: Path):
    doc = ezdxf.new(dxfversion="R2010")

    for name, color, lw in [
        ("WALLS", 2, 50),
        ("DOORS", 1, 25),
        ("WINDOWS", 5, 25),
        ("LABELS", 4, 13),
    ]:
        if name not in doc.layers:
            doc.layers.new(name, dxfattribs={"color": color, "lineweight": lw})

    msp = doc.modelspace()
    scale = C.DXF_SCALE

    for i in range(n_walls):
        p1 = pts_m[i] * scale
        p2 = pts_m[(i + 1) % n_walls] * scale
        msp.add_line((p1[0], p1[1]), (p2[0], p2[1]), dxfattribs={"layer": "WALLS", "lineweight": 50})

    for op in mapped_openings:
        cp = op["center"] * scale
        wall_dir = op["wall_dir"]
        normal = op["normal"]
        open_width = op["width"] * scale
        half_width = open_width / 2

        open_start = cp - wall_dir * half_width
        open_end = cp + wall_dir * half_width

        if op["type"] == "door":
            msp.add_line(
                (open_start[0], open_start[1]),
                (open_start[0] + normal[0] * open_width, open_start[1] + normal[1] * open_width),
                dxfattribs={"layer": "DOORS"},
            )
            ang0 = np.degrees(np.arctan2(wall_dir[1], wall_dir[0]))
            msp.add_arc(
                (open_start[0], open_start[1], 0),
                open_width,
                ang0,
                ang0 + 90,
                dxfattribs={"layer": "DOORS"},
            )
        else:
            thickness = C.WALL_THICKNESS * scale
            for off in (-thickness / 2, 0, thickness / 2):
                msp.add_line(
                    (open_start[0] + normal[0] * off, open_start[1] + normal[1] * off),
                    (open_end[0] + normal[0] * off, open_end[1] + normal[1] * off),
                    dxfattribs={"layer": "WINDOWS"},
                )

        msp.add_text(
            op["tag"],
            dxfattribs={"layer": "LABELS", "height": scale * 0.12},
        ).set_placement(
            (cp[0], cp[1]),
            align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
        )

    doc.saveas(str(out_dxf))


def run_pipeline(image_bytes: bytes, job_id: str) -> dict:
    out_png = C.OUTPUT_DIR / f"{job_id}_floorplan.png"
    out_dxf = C.OUTPUT_DIR / f"{job_id}_floorplan.dxf"
    out_layout_debug = C.OUTPUT_DIR / f"{job_id}_layout_debug.png"
    out_openings_debug = C.OUTPUT_DIR / f"{job_id}_openings_debug.png"

    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cor, bon, img_np, img_pil_resized = run_horizonnet(img_pil)
    width, height = img_pil_resized.size

    pts_m, n_walls, room_center, corner_pixels, smooth = build_floorplan_polygon(cor, width)
    dw_dets = detect_doors_windows(img_pil_resized)
    mapped_openings = map_openings(dw_dets, pts_m, n_walls, room_center, width, height, corner_pixels)

    save_layout_debug(pts_m, room_center, corner_pixels, width, out_layout_debug)
    save_openings_debug_overlay(img_np, mapped_openings, out_openings_debug)
    render_floorplan(pts_m, n_walls, mapped_openings, out_png, out_dxf)

    doors = [o for o in mapped_openings if o["type"] == "door"]
    windows = [o for o in mapped_openings if o["type"] == "window"]

    return {
        "job_id": job_id,
        "walls": n_walls,
        "doors": len(doors),
        "windows": len(windows),
        "openings": [
            {
                "type": o["type"],
                "tag": o["tag"],
                "conf": o["conf"],
                "wall": o["wall_i"],
                "frac": round(float(o["frac"]), 3),
                "raw_class": o["raw_class"],
                "class_id": o["class_id"],
                "pano_xy": o["pano_xy"],
            }
            for o in mapped_openings
        ],
        "room_size_m": {
            "width": round(float(np.ptp(pts_m[:, 0])), 2),
            "depth": round(float(np.ptp(pts_m[:, 1])), 2),
        },
        "png_path": str(out_png),
        "dxf_path": str(out_dxf),
        "layout_debug_png_path": str(out_layout_debug),
        "openings_debug_png_path": str(out_openings_debug),
    }