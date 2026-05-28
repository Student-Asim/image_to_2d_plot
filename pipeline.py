import io
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.signal import find_peaks
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

import config as C


log = logging.getLogger("pipeline")
_horizonnet_model = None
_furniture_model = None


def load_models():
    global _horizonnet_model, _furniture_model

    sys.path.append(str(C.HORIZONNET_DIR))
    from model import HorizonNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HorizonNet(backbone="resnet50", use_rnn=True).to(device)
    ckpt = torch.load(C.HORIZONNET_WEIGHTS, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    model.eval()
    _horizonnet_model = model
    log.info("HorizonNet loaded on %s", device)

    _furniture_model = YOLO(C.YOLO_MODEL)
    log.info("Furniture YOLO loaded: %s", C.YOLO_MODEL)


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
            timeout=(C.RF_CONNECT_TIMEOUT, C.RF_READ_TIMEOUT),
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

    all_raw = [
    d for d in all_raw
    if d["conf"] >= C.DW_CLASS_CONF.get(d["class"], C.CONF_THRESH)
    ]

    final = []
    for cls_name in sorted(set(d["class"] for d in all_raw)):
        cls_dets = [d for d in all_raw if d["class"] == cls_name]
        final.extend(_nms(cls_dets, C.NMS_IOU, pano_w=pano_w))

    for d in final:
        d["x"] = d["cx"]
        d["y"] = d["cy"]
        d["confidence"] = d["conf"]

    return final


def detect_furniture(img_pil: Image.Image):
    if _furniture_model is None:
        raise RuntimeError("Furniture model not loaded.")

    img_np = np.array(img_pil)
    results = _furniture_model.predict(
        source=img_np,
        conf=min(C.FURNITURE_CONF, 0.10),
        imgsz=getattr(C, "FURNITURE_IMGSZ", 1024),
        verbose=False,
    )

    class_conf = getattr(C, "FURNITURE_CLASS_CONF", {
        "bed": 0.18,
        "chair": 0.10,
        "sofa": 0.14,
        "couch": 0.14,
        "dining table": 0.12,
        "bench": 0.12,
        "tv": 0.12,
    })

    dets = []
    for r in results:
        names = r.names
        if r.boxes is None:
            continue

        for b in r.boxes:
            cls_id = int(b.cls.item())
            cls_name = names.get(cls_id, str(cls_id))
            mapped_name = C.FURNITURE_CLASSES.get(cls_name)
            if mapped_name is None:
                continue

            conf = float(b.conf.item())
            min_conf = class_conf.get(cls_name, C.FURNITURE_CONF)
            if conf < min_conf:
                continue

            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            dets.append({
                "type": mapped_name,
                "label": mapped_name.title(),
                "raw_class": cls_name,
                "conf": conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float((x1 + x2) / 2.0),
                "cy": float((y1 + y2) / 2.0),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "area_px": float((x2 - x1) * (y2 - y1)),
            })

    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    return dets


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


def _furniture_size_m(furn_type, room_w, room_h):
    base = {
        "bed": (1.55, 1.15),
        "sofa": (1.15, 0.65),
        "table": (0.85, 0.55),
        "chair": (0.40, 0.40),
    }.get(furn_type, (0.60, 0.50))

    max_w = room_w * 0.35
    max_h = room_h * 0.26

    w = min(base[0], max_w)
    h = min(base[1], max_h)

    if furn_type == "chair":
        w = min(w, room_w * 0.12)
        h = min(h, room_h * 0.12)
    elif furn_type == "table":
        w = min(w, room_w * 0.22)
        h = min(h, room_h * 0.18)
    elif furn_type == "sofa":
        w = min(w, room_w * 0.28)
        h = min(h, room_h * 0.18)
    elif furn_type == "bed":
        w = min(w, room_w * 0.32)
        h = min(h, room_h * 0.24)

    return (max(w, 0.28), max(h, 0.28))


def _rot_points(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def _rot_rect(center, w, h, angle):
    pts = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2],
    ], dtype=float)
    return _rot_points(pts, angle) + np.asarray(center, dtype=float)


def _opening_rect(op, pad=0.04):
    cp = np.asarray(op["center"], dtype=float)
    w = float(op["width"]) + pad
    h = max(C.WALL_THICKNESS * 1.35, 0.18)
    ang = float(np.arctan2(op["wall_dir"][1], op["wall_dir"][0]))
    return _rot_rect(cp, w, h, ang)


def _furniture_polygon(center, size, angle):
    rect = _rot_rect(center, size[0], size[1], angle)
    return Polygon(rect)


def _inside_room(center, size, angle, room_poly, wall_gap=0.04):
    poly = _furniture_polygon(center, size, angle)
    inner = room_poly.buffer(-wall_gap)
    if inner.is_empty:
        inner = room_poly
    return inner.covers(poly)


def _overlaps_existing(center, size, angle, placed_furniture, opening_polys, gap=0.03):
    poly = _furniture_polygon(center, size, angle).buffer(gap)
    for fp in placed_furniture:
        if poly.intersects(fp["poly"].buffer(gap)):
            return True
    for op in opening_polys:
        if poly.intersects(op.buffer(gap)):
            return True
    return False


def _find_valid_furniture_placement(base_center, size, angle, room_poly, placed_furniture, opening_polys):
    scales = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]
    offsets = [
        (0.0, 0.0),
        (0.10, 0.0), (-0.10, 0.0),
        (0.0, 0.10), (0.0, -0.10),
        (0.18, 0.0), (-0.18, 0.0),
        (0.0, 0.18), (0.0, -0.18),
        (0.25, 0.0), (-0.25, 0.0),
        (0.0, 0.25), (0.0, -0.25),
        (0.15, 0.15), (-0.15, 0.15),
        (0.15, -0.15), (-0.15, -0.15),
        (0.30, 0.15), (-0.30, 0.15),
        (0.30, -0.15), (-0.30, -0.15),
    ]

    for scale in scales:
        test_size = (size[0] * scale, size[1] * scale)
        for dx, dy in offsets:
            center = np.array([base_center[0] + dx, base_center[1] + dy], dtype=float)
            if not _inside_room(center, test_size, angle, room_poly):
                continue
            if _overlaps_existing(center, test_size, angle, placed_furniture, opening_polys):
                continue
            return center, test_size

    return None, None


def map_furniture(furniture_dets, pts_m, n_walls, room_center, img_w, img_h, mapped_openings):
    mapped = []
    room_poly = Polygon(pts_m)
    opening_polys = [Polygon(_opening_rect(op)) for op in mapped_openings]

    room_w = float(np.ptp(pts_m[:, 0]))
    room_h = float(np.ptp(pts_m[:, 1]))

    for i, det in enumerate(sorted(furniture_dets, key=lambda d: -d["conf"]), start=1):
        angle = _pano_x_to_angle(det["cx"], img_w)
        hit = _find_hit_wall(room_center, angle, pts_m, n_walls)
        if hit is None:
            log.info("skip furniture %s: no wall hit", det["label"])
            continue

        _, frac, wall_i = hit
        p1 = pts_m[wall_i]
        p2 = pts_m[(wall_i + 1) % n_walls]
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1e-6:
            log.info("skip furniture %s: tiny wall", det["label"])
            continue

        frac = float(np.clip(frac, 0.12, 0.88))
        wall_dir = wall_vec / wall_len
        normal = np.array([-wall_dir[1], wall_dir[0]], dtype=float)

        probe = p1 + frac * wall_vec + normal * 0.30
        if not room_poly.covers(Point(probe[0], probe[1])):
            normal = -normal

        size = _furniture_size_m(det["type"], room_w, room_h)

        rel_y = np.clip(det["cy"] / float(img_h), 0.05, 0.98)
        inward_offset = 0.35 + (1.0 - rel_y) * 0.55

        if det["type"] == "chair":
            inward_offset *= 0.75
        elif det["type"] == "table":
            inward_offset *= 0.85
        elif det["type"] == "bed":
            inward_offset *= 1.05

        base_center = p1 + frac * wall_vec + normal * inward_offset
        furn_angle = float(np.arctan2(wall_dir[1], wall_dir[0]))

        center, final_size = _find_valid_furniture_placement(
            base_center=base_center,
            size=size,
            angle=furn_angle,
            room_poly=room_poly,
            placed_furniture=mapped,
            opening_polys=opening_polys,
        )

        if center is None:
            log.info(
                "skip furniture %s: no valid placement, conf=%.3f",
                det["label"], det["conf"]
            )
            continue

        poly = _furniture_polygon(center, final_size, furn_angle)
        mapped.append({
            "tag": f"F{i}",
            "type": det["type"],
            "label": det["label"],
            "center": center,
            "angle": furn_angle,
            "size": final_size,
            "conf": float(det["conf"]),
            "wall_i": int(wall_i),
            "pano_xy": [float(det["cx"]), float(det["cy"])],
            "poly": poly,
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
            f"{op['tag']} {op['type']} -> wall {op['wall_i'] + 1} {op['frac']:.2f}",
            (max(8, x + 8), max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), vis)


def save_furniture_debug_overlay(img_np, furniture_dets, out_path: Path):
    vis = img_np.copy()
    h, w = vis.shape[:2]
    colors = {
        "bed": (120, 80, 220),
        "sofa": (50, 170, 90),
        "table": (220, 140, 40),
        "chair": (200, 80, 80),
    }

    for det in furniture_dets:
        x1 = int(np.clip(det["x1"], 0, w - 1))
        y1 = int(np.clip(det["y1"], 0, h - 1))
        x2 = int(np.clip(det["x2"], 0, w - 1))
        y2 = int(np.clip(det["y2"], 0, h - 1))
        color = colors.get(det["type"], (180, 180, 80))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (int(det["cx"]), int(det["cy"])), 4, color, -1)
        cv2.putText(
            vis,
            f"{det['label']} {det['conf']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), vis)


def _draw_wall_segment(ax, p1, p2, openings):
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return

    wall_dir = vec / length
    normal = np.array([-wall_dir[1], wall_dir[0]], dtype=float)
    thickness = max(C.WALL_THICKNESS, 0.18)

    cuts = sorted(
        (
            np.clip(op["frac"] - op["width"] / 2 / length, 0.0, 1.0),
            np.clip(op["frac"] + op["width"] / 2 / length, 0.0, 1.0),
        )
        for op in openings
    )

    segments = []
    prev = 0.0
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

        ax.add_patch(plt.Polygon(
            poly,
            closed=True,
            fc="black",
            ec="black",
            lw=0.5,
            joinstyle="miter",
            zorder=3
        ))


def _draw_door(ax, op):
    p1 = op["p1"]
    p2 = op["p2"]
    wall_len = op["wall_len"]
    wall_dir = op["wall_dir"]
    normal = op["normal"]

    cut_start = np.clip(op["frac"] - op["width"] / 2 / wall_len, 0.0, 1.0)
    cut_end = np.clip(op["frac"] + op["width"] / 2 / wall_len, 0.0, 1.0)

    open_start = p1 + cut_start * (p2 - p1)
    open_end = p1 + cut_end * (p2 - p1)
    open_width = np.linalg.norm(open_end - open_start)

    if open_width < 1e-6:
        return

    # hinge side = start point
    hinge = open_start

    # closed leaf direction runs along wall opening
    closed_tip = open_end

    # open leaf direction swings inward/outward along wall normal
    open_tip = hinge + normal * open_width

    # optional faint closed-door guide
    ax.plot(
        [hinge[0], closed_tip[0]],
        [hinge[1], closed_tip[1]],
        color="#b8b8b8",
        lw=0.9,
        ls="-",
        zorder=5,
    )

    # main open door leaf
    ax.plot(
        [hinge[0], open_tip[0]],
        [hinge[1], open_tip[1]],
        color="#2d2d2d",
        lw=1.5,
        solid_capstyle="round",
        zorder=7,
    )

    # hinge point
    ax.scatter(
        [hinge[0]],
        [hinge[1]],
        s=10,
        c="#2d2d2d",
        zorder=8
    )

    # swing arc from closed position to open position
    base = np.arctan2(wall_dir[1], wall_dir[0])
    theta = np.linspace(0, np.pi / 2, 80)

    arc = np.c_[
        hinge[0] + np.cos(base + theta) * open_width,
        hinge[1] + np.sin(base + theta) * open_width,
    ]
    ax.plot(
        arc[:, 0],
        arc[:, 1],
        color="#6f6f6f",
        lw=1.0,
        ls="-",
        zorder=6
    )

def _draw_window(ax, op):
    p1 = op["p1"]
    p2 = op["p2"]
    wall_len = op["wall_len"]
    normal = op["normal"]

    cut_start = np.clip(op["frac"] - op["width"] / 2 / wall_len, 0.0, 1.0)
    cut_end = np.clip(op["frac"] + op["width"] / 2 / wall_len, 0.0, 1.0)

    open_start = p1 + cut_start * (p2 - p1)
    open_end = p1 + cut_end * (p2 - p1)

    thickness = max(C.WALL_THICKNESS, 0.18)
    depth = thickness * 0.42
    cap = thickness * 0.18

    # two parallel frame lines
    for off in (-depth, depth):
        wx1 = open_start + normal * off
        wx2 = open_end + normal * off
        ax.plot(
            [wx1[0], wx2[0]],
            [wx1[1], wx2[1]],
            color="white",
            lw=2.1,
            solid_capstyle="butt",
            zorder=6
        )

    # middle glass/opening line
    ax.plot(
        [open_start[0], open_end[0]],
        [open_start[1], open_end[1]],
        color="#222222",
        lw=0.9,
        solid_capstyle="butt",
        zorder=7
    )

    # small end caps
    for pt in (open_start, open_end):
        c1 = pt - normal * cap
        c2 = pt + normal * cap
        ax.plot(
            [c1[0], c2[0]],
            [c1[1], c2[1]],
            color="#222222",
            lw=0.9,
            zorder=7
        )
def _draw_furniture(ax, furn):
    t = furn["type"]
    center = np.asarray(furn["center"], dtype=float)
    angle = furn["angle"]
    w, h = furn["size"]

    label = furn.get("label", t).lower()

    edge = "#8a8a8a"
    edge_dark = "#666666"
    fill = "#efefef"
    fill_mid = "#dddddd"
    fill_dark = "#cfcfcf"

    if t == "bed":
        frame = _rot_rect(center, w, h, angle)
        ax.add_patch(plt.Polygon(
            frame, closed=True, fc="#f3f3f3", ec=edge_dark, lw=0.9, zorder=2
        ))

        mattress_center = _rot_points(np.array([[0.0, -h * 0.01]]), angle)[0] + center
        mattress = _rot_rect(mattress_center, w * 0.90, h * 0.88, angle)
        ax.add_patch(plt.Polygon(
            mattress, closed=True, fc="#fbfbfb", ec=edge, lw=0.5, zorder=3
        ))

        hb_center = _rot_points(np.array([[0.0, h * 0.44]]), angle)[0] + center
        headboard = _rot_rect(hb_center, w * 0.95, h * 0.09, angle)
        ax.add_patch(plt.Polygon(
            headboard, closed=True, fc=fill_dark, ec=edge, lw=0.45, zorder=4
        ))

        local1 = np.array([-w * 0.22, h * 0.26])
        local2 = np.array([ w * 0.22, h * 0.26])
        pcenters = _rot_points(np.array([local1, local2]), angle) + center
        for pc in pcenters:
            pillow = _rot_rect(pc, w * 0.24, h * 0.16, angle)
            ax.add_patch(plt.Polygon(
                pillow, closed=True, fc="white", ec=edge, lw=0.45, zorder=5
            ))

        a = _rot_points(np.array([[-w * 0.34, -h * 0.03]]), angle)[0] + center
        b = _rot_points(np.array([[ w * 0.34, -h * 0.03]]), angle)[0] + center
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#b5b5b5", lw=0.45, zorder=5)

    elif t == "chair":
        seat_w = w * 0.62
        seat_h = h * 0.62

        seat = _rot_rect(center, seat_w, seat_h, angle)
        ax.add_patch(plt.Polygon(
            seat, closed=True, fc="#f8f8f8", ec=edge_dark, lw=0.75, zorder=3
        ))

        back_center = _rot_points(np.array([[0.0, seat_h * 0.44]]), angle)[0] + center
        back = _rot_rect(back_center, seat_w * 0.92, seat_h * 0.18, angle)
        ax.add_patch(plt.Polygon(
            back, closed=True, fc=fill_dark, ec=edge, lw=0.4, zorder=4
        ))

        left_arm_center = _rot_points(np.array([[-seat_w * 0.44, 0.0]]), angle)[0] + center
        right_arm_center = _rot_points(np.array([[ seat_w * 0.44, 0.0]]), angle)[0] + center

        for ac in [left_arm_center, right_arm_center]:
            arm = _rot_rect(ac, seat_w * 0.14, seat_h * 0.78, angle)
            ax.add_patch(plt.Polygon(
                arm, closed=True, fc=fill_mid, ec=edge, lw=0.35, zorder=4
            ))

        cushion = _rot_rect(center, seat_w * 0.52, seat_h * 0.42, angle)
        ax.add_patch(plt.Polygon(
            cushion, closed=True, fc="white", ec="#b8b8b8", lw=0.3, zorder=5
        ))

    elif t == "sofa":
        rect = _rot_rect(center, w, h, angle)
        ax.add_patch(plt.Polygon(
            rect, closed=True, fc=fill_mid, ec=edge_dark, lw=0.9, zorder=2
        ))

        inner = _rot_rect(center, w * 0.74, h * 0.48, angle)
        ax.add_patch(plt.Polygon(
            inner, closed=True, fc="#f8f8f8", ec=edge, lw=0.5, zorder=3
        ))

        back_center = _rot_points(np.array([[0.0, h * 0.30]]), angle)[0] + center
        back = _rot_rect(back_center, w * 0.84, h * 0.15, angle)
        ax.add_patch(plt.Polygon(
            back, closed=True, fc=fill_dark, ec=edge, lw=0.5, zorder=4
        ))

        arm1_center = _rot_points(np.array([[-w * 0.42, 0.0]]), angle)[0] + center
        arm2_center = _rot_points(np.array([[ w * 0.42, 0.0]]), angle)[0] + center
        for ac in [arm1_center, arm2_center]:
            arm = _rot_rect(ac, w * 0.10, h * 0.70, angle)
            ax.add_patch(plt.Polygon(
                arm, closed=True, fc=fill_dark, ec=edge, lw=0.5, zorder=4
            ))

    elif t == "table":
        rect = _rot_rect(center, w, h, angle)
        ax.add_patch(plt.Polygon(
            rect, closed=True, fc="#f4f4f4", ec=edge_dark, lw=0.9, zorder=2
        ))

    else:
        rect = _rot_rect(center, w, h, angle)
        ax.add_patch(plt.Polygon(
            rect, closed=True, fc="#e8e8e8", ec=edge_dark, lw=0.8, zorder=2
        ))

    ax.text(
        center[0],
        center[1] - max(h, w) * 0.42,
        label,
        fontsize=7,
        color="#303030",
        ha="center",
        va="top",
        zorder=8,
        bbox=dict(
            boxstyle="round,pad=0.14",
            fc="white",
            ec="none",
            alpha=0.80
        )
    )

def _draw_dimension(ax, p1, p2, room_center, text):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    room_center = np.asarray(room_center, dtype=float)

    mid = (p1 + p2) / 2.0
    wall_vec = p2 - p1
    wall_len = np.linalg.norm(wall_vec)
    if wall_len < 1e-6:
        return

    wall_dir = wall_vec / wall_len
    normal = np.array([-wall_dir[1], wall_dir[0]], dtype=float)

    # make label go outside room
    if np.dot(normal, mid - room_center) < 0:
        normal = -normal

    offset = 0.22
    tick = 0.08

    dp1 = p1 + normal * offset
    dp2 = p2 + normal * offset
    tm = mid + normal * (offset + 0.02)

    ax.plot([p1[0], dp1[0]], [p1[1], dp1[1]], color="#7a7a7a", lw=0.6, zorder=6)
    ax.plot([p2[0], dp2[0]], [p2[1], dp2[1]], color="#7a7a7a", lw=0.6, zorder=6)
    ax.plot([dp1[0], dp2[0]], [dp1[1], dp2[1]], color="#7a7a7a", lw=0.8, zorder=6)

    t1a = dp1 - wall_dir * tick / 2
    t1b = dp1 + wall_dir * tick / 2
    t2a = dp2 - wall_dir * tick / 2
    t2b = dp2 + wall_dir * tick / 2

    ax.plot([t1a[0], t1b[0]], [t1a[1], t1b[1]], color="#7a7a7a", lw=0.8, zorder=6)
    ax.plot([t2a[0], t2b[0]], [t2a[1], t2b[1]], color="#7a7a7a", lw=0.8, zorder=6)

    ax.text(
        tm[0], tm[1], text,
        fontsize=8,
        color="#444444",
        ha="center",
        va="center",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
    ) 

def render_floorplan(pts_m, n_walls, mapped_openings, mapped_furniture, out_png: Path, out_dxf: Path):
    pad = 0.95
    xlim = (pts_m[:, 0].min() - pad, pts_m[:, 0].max() + pad)
    ylim = (pts_m[:, 1].min() - pad, pts_m[:, 1].max() + pad)

    fig, ax = plt.subplots(figsize=(8.5, 8.5), facecolor="white")
    ax.set_facecolor("white")

    ax.add_patch(plt.Polygon(
        pts_m,
        closed=True,
        fc="#f2f2ef",
        ec="none",
        zorder=1
    ))

    for furn in mapped_furniture:
        _draw_furniture(ax, furn)

    openings_by_wall = defaultdict(list)
    for op in mapped_openings:
        openings_by_wall[op["wall_i"]].append(op)

    for i in range(n_walls):
        _draw_wall_segment(ax, pts_m[i], pts_m[(i + 1) % n_walls], openings_by_wall[i])

    ax.add_patch(plt.Polygon(
        pts_m,
        closed=True,
        fill=False,
        edgecolor="black",
        linewidth=2.2,
        joinstyle="miter",
        zorder=4
    ))

    for op in mapped_openings:
        if op["type"] == "door":
            _draw_door(ax, op)
        else:
            _draw_window(ax, op)

    room_center = pts_m.mean(axis=0)
    room_area = 0.5 * abs(
        np.dot(pts_m[:, 0], np.roll(pts_m[:, 1], -1)) -
        np.dot(pts_m[:, 1], np.roll(pts_m[:, 0], -1))
    )

    for i in range(n_walls):
        p1 = pts_m[i]
        p2 = pts_m[(i + 1) % n_walls]
        wall_len = np.linalg.norm(p2 - p1)
        _draw_dimension(ax, p1, p2, room_center, f"{wall_len:.2f} m")

    ax.text(
        room_center[0],
        room_center[1],
        f"Room\n{room_area:.2f} sq m",
        ha="center",
        va="center",
        fontsize=11,
        color="#2b2b2b",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc=(1, 1, 1, 0.82),
            ec="none"
        )
    )

    for op in mapped_openings:
        cp = op["center"]
        ax.text(
            cp[0],
            cp[1],
            op["tag"],
            fontsize=7,
            color="#202020",
            ha="center",
            va="center",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.90)
        )

    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(str(out_png), dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    _export_dxf(pts_m, n_walls, mapped_openings, mapped_furniture, out_dxf)


def _export_dxf(pts_m, n_walls, mapped_openings, mapped_furniture, out_dxf: Path):
    doc = ezdxf.new(dxfversion="R2010")

    for name, color, lw in [
        ("WALLS", 2, 50),
        ("DOORS", 1, 25),
        ("WINDOWS", 5, 25),
        ("LABELS", 4, 13),
        ("DIMS", 8, 13),
        ("FURNITURE", 7, 13),
    ]:
        if name not in doc.layers:
            doc.layers.new(name, dxfattribs={"color": color, "lineweight": lw})

    msp = doc.modelspace()
    scale = C.DXF_SCALE

    for furn in mapped_furniture:
        cp = np.asarray(furn["center"]) * scale
        w, h = furn["size"]
        w *= scale
        h *= scale

        if furn["type"] == "chair":
            msp.add_circle((cp[0], cp[1]), radius=min(w, h) * 0.33, dxfattribs={"layer": "FURNITURE"})
        else:
            pts = _rot_rect(np.array([0.0, 0.0]), w, h, furn["angle"]) + cp
            pts = [(float(p[0]), float(p[1])) for p in pts]
            msp.add_lwpolyline(pts + [pts[0]], dxfattribs={"layer": "FURNITURE"})

        msp.add_text(
            furn["label"],
            dxfattribs={"layer": "LABELS", "height": scale * 0.10},
        ).set_placement(
            (cp[0], cp[1]),
            align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
        )

    for i in range(n_walls):
        p1 = pts_m[i] * scale
        p2 = pts_m[(i + 1) % n_walls] * scale
        msp.add_line(
            (p1[0], p1[1]),
            (p2[0], p2[1]),
            dxfattribs={"layer": "WALLS", "lineweight": 50}
        )

    room_center = pts_m.mean(axis=0) * scale

    for i in range(n_walls):
        p1 = pts_m[i] * scale
        p2 = pts_m[(i + 1) % n_walls] * scale
        vec = p2 - p1
        wall_len = np.linalg.norm(vec)
        if wall_len < 1e-6:
            continue

        wall_dir = vec / wall_len
        normal = np.array([-wall_dir[1], wall_dir[0]], dtype=float)
        mid = (p1 + p2) / 2.0
        if np.dot(normal, room_center - mid) > 0:
            normal = -normal

        offset = 0.28 * scale
        q1 = p1 + normal * offset
        q2 = p2 + normal * offset
        msp.add_line((q1[0], q1[1]), (q2[0], q2[1]), dxfattribs={"layer": "DIMS"})

    for op in mapped_openings:
        cp = op["center"] * scale
        wall_dir = op["wall_dir"]
        normal = op["normal"]
        open_width = op["width"] * scale
        half_width = open_width / 2.0

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
            thickness = max(C.WALL_THICKNESS * scale, 0.18 * scale)
            for off in (-thickness / 2.0, 0.0, thickness / 2.0):
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
    out_furniture_debug = C.OUTPUT_DIR / f"{job_id}_furniture_debug.png"

    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cor, bon, img_np, img_pil_resized = run_horizonnet(img_pil)
    width, height = img_pil_resized.size

    pts_m, n_walls, room_center, corner_pixels, smooth = build_floorplan_polygon(cor, width)
    dw_dets = detect_doors_windows(img_pil_resized)
    mapped_openings = map_openings(dw_dets, pts_m, n_walls, room_center, width, height, corner_pixels)

    furniture_dets = detect_furniture(img_pil_resized)
    mapped_furniture = map_furniture(
        furniture_dets,
        pts_m,
        n_walls,
        room_center,
        width,
        height,
        mapped_openings,
    )

    log.info("Furniture detections raw: %s", [
        (d["type"], round(d["conf"], 3), round(d["cx"], 1), round(d["cy"], 1))
        for d in furniture_dets
    ])
    log.info("Furniture mapped: %s", [
        (
            f["type"],
            f["label"],
            [round(x, 2) for x in f["center"]],
            [round(s, 2) for s in f["size"]],
        )
        for f in mapped_furniture
    ])

    save_layout_debug(pts_m, room_center, corner_pixels, width, out_layout_debug)
    save_openings_debug_overlay(img_np, mapped_openings, out_openings_debug)
    save_furniture_debug_overlay(img_np, furniture_dets, out_furniture_debug)
    render_floorplan(pts_m, n_walls, mapped_openings, mapped_furniture, out_png, out_dxf)

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
        "furniture": [
            {
                "tag": f["tag"],
                "type": f["type"],
                "label": f["label"],
                "conf": round(float(f["conf"]), 3),
                "wall": f["wall_i"],
                "size_m": [round(float(f["size"][0]), 2), round(float(f["size"][1]), 2)],
                "pano_xy": [round(float(f["pano_xy"][0]), 1), round(float(f["pano_xy"][1]), 1)],
            }
            for f in mapped_furniture
        ],
        "room_size_m": {
            "width": round(float(np.ptp(pts_m[:, 0])), 2),
            "depth": round(float(np.ptp(pts_m[:, 1])), 2),
            "area": round(float(
                0.5 * abs(
                    np.dot(pts_m[:, 0], np.roll(pts_m[:, 1], -1)) -
                    np.dot(pts_m[:, 1], np.roll(pts_m[:, 0], -1))
                )
            ), 2),
        },
        "png_path": str(out_png),
        "dxf_path": str(out_dxf),
        "layout_debug_png_path": str(out_layout_debug),
        "openings_debug_png_path": str(out_openings_debug),
        "furniture_debug_png_path": str(out_furniture_debug),
    }
