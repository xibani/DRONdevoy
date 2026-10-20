"""Odometría visual monocular de DOS fotogramas, con visualizaciones didácticas.

Uso:
  python VO_2_frames_visual.py video.mp4 --first 1 --second 4
  python VO_2_frames_visual.py video.mp4 --first 1 --second 20 --show

Dependencias: numpy, opencv-python, matplotlib.

Se implementan en NumPy: normalización de Hartley, 8 puntos, RANSAC,
error de Sampson, descomposición de E, selección por quiralidad y triangulación.
OpenCV se usa para vídeo, SIFT, matching y dibujos, NO para findEssentialMat,
recoverPose ni triangulatePoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"No se puede leer el fotograma {idx}")
    return frame


def homogeneous(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.ones(len(x))])


def from_homogeneous(xh: np.ndarray) -> np.ndarray:
    return xh[:, :2] / xh[:, 2:3]


def normalized_image_points(x: np.ndarray, K: np.ndarray) -> np.ndarray:
    return from_homogeneous((np.linalg.inv(K) @ homogeneous(x).T).T)


def hartley_normalization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = x.mean(axis=0)
    radius = np.linalg.norm(x - center, axis=1).mean()
    if radius < 1e-12:
        raise ValueError("Configuración degenerada de puntos")
    scale = np.sqrt(2.0) / radius
    T = np.array(
        [[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1.0]]
    )
    return from_homogeneous((T @ homogeneous(x).T).T), T


def essential_eight_points(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Método lineal de 8 puntos. x1/x2 son Nx2 en coordenadas calibradas K^-1."""
    if len(x1) < 8:
        raise ValueError("Necesitamos al menos ocho correspondencias")
    p1, T1 = hartley_normalization(x1)
    p2, T2 = hartley_normalization(x2)
    a, b = p1[:, 0], p1[:, 1]
    c, d = p2[:, 0], p2[:, 1]
    A = np.column_stack([c * a, c * b, c, d * a, d * b, d, a, b, np.ones(len(a))])
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    E0 = Vt[-1].reshape(3, 3)
    # Restringir rango 2 (antes de desnormalizar)
    U, S, Vt = np.linalg.svd(E0)
    E0 = U @ np.diag([S[0], S[1], 0.0]) @ Vt
    E = T2.T @ E0 @ T1
    # Matriz esencial: dos valores singulares iguales, tercero nulo
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    return E / np.linalg.norm(E)


def sampson_error(E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    a, b = homogeneous(x1), homogeneous(x2)
    Ea = (E @ a.T).T
    Etb = (E.T @ b.T).T
    residual = np.einsum("ij,ij->i", b, Ea)
    den = Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2
    return residual**2 / np.maximum(den, 1e-15)


def ransac_essential(
    x1: np.ndarray,
    x2: np.ndarray,
    threshold: float,
    iterations: int = 2500,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(x1), dtype=bool)
    for _ in range(iterations):
        ids = rng.choice(len(x1), size=8, replace=False)
        try:
            E = essential_eight_points(x1[ids], x2[ids])
        except (np.linalg.LinAlgError, ValueError):
            continue
        current = sampson_error(E, x1, x2) < threshold**2
        if current.sum() > best_mask.sum():
            best_mask = current
    if best_mask.sum() < 12:
        raise RuntimeError("RANSAC: menos de doce correspondencias consistentes")
    E = essential_eight_points(x1[best_mask], x2[best_mask])
    # Refinar una vez con el nuevo conjunto de inliers
    refined = sampson_error(E, x1, x2) < threshold**2
    if refined.sum() >= 12:
        best_mask = refined
        E = essential_eight_points(x1[best_mask], x2[best_mask])
    # La máscara devuelta debe corresponder con la E FINAL
    final_mask = sampson_error(E, x1, x2) < threshold**2
    if final_mask.sum() >= 8:
        best_mask = final_mask
    return E, best_mask


def triangulate_point(
    P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray | None:
    """Triangulación DLT, para coordenadas de imagen normalizadas."""
    A = np.array(
        [
            x1[0] * P1[2] - P1[0],
            x1[1] * P1[2] - P1[1],
            x2[0] * P2[2] - P2[0],
            x2[1] * P2[2] - P2[1],
        ]
    )
    _, _, Vt = np.linalg.svd(A)
    if abs(Vt[-1, 3]) < 1e-10:
        return None
    X = Vt[-1, :3] / Vt[-1, 3]
    return X if np.isfinite(X).all() else None


def triangulate_all(
    R: np.ndarray, t: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    P1 = np.column_stack([np.eye(3), np.zeros(3)])
    P2 = np.column_stack([R, t])
    X = np.full((len(x1), 3), np.nan)
    for i in range(len(x1)):
        point = triangulate_point(P1, P2, x1[i], x2[i])
        if point is not None:
            X[i] = point
    return X


def positive_depth_mask(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    depth1 = X[:, 2]
    depth2 = (R @ X.T).T[:, 2] + t[2]
    return np.isfinite(X).all(axis=1) & (depth1 > 0) & (depth2 > 0)


def decompose_pose(E: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """Selecciona entre (R1, +/-t), (R2, +/-t) por profundidad positiva."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotations = [U @ W @ Vt, U @ W.T @ Vt]
    direction = U[:, 2]
    best_score = -1
    best_pose = None
    sample_ids = np.linspace(0, len(x1) - 1, min(100, len(x1)), dtype=int)
    for R in rotations:
        for t in (direction, -direction):
            X = triangulate_all(R, t, x1[sample_ids], x2[sample_ids])
            score = int(positive_depth_mask(X, R, t).sum())
            if score > best_score:
                best_score = score
                best_pose = (R.copy(), t.copy())
    if best_pose is None:
        raise RuntimeError("No se ha podido recuperar la pose")
    return (*best_pose, best_score, len(sample_ids))


def euler_zyx_deg(R: np.ndarray) -> tuple[float, float, float]:
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return tuple(np.degrees([roll, pitch, yaw]))


def project_points(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray
) -> np.ndarray:
    coords = (R @ X.T).T + t
    pix = (K @ coords.T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return pix[:, :2] / pix[:, 2:3]


def triangulation_diagnostics(K, R, t, X, x1, x2, uv1, uv2, min_parallax_deg):
    valid = positive_depth_mask(X, R, t)
    C2 = -R.T @ t
    # Rayos de los dos centros de cámara hacia el punto triangulado.
    v1, v2 = X.copy(), X - C2
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cosines = np.einsum("ij,ij->i", v1, v2) / np.maximum(n1 * n2, 1e-15)
    parallax = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    proj1 = project_points(K, np.eye(3), np.zeros(3), X)
    proj2 = project_points(K, R, t, X)
    reproj = (
        np.linalg.norm(proj1 - uv1, axis=1) + np.linalg.norm(proj2 - uv2, axis=1)
    ) / 2
    reliable = (
        valid & np.isfinite(reproj) & (parallax >= min_parallax_deg) & (reproj < 3.0)
    )
    return {
        "positive_mask": valid,
        "reliable_mask": reliable,
        "parallax_deg": parallax,
        "reprojection_px": reproj,
    }


def pick_indices(n: int, maximum: int, seed: int = 5) -> np.ndarray:
    if n <= maximum:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=maximum, replace=False))


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_cv(path: Path, img: np.ndarray):
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"No se pudo guardar {path}")


def draw_matches(
    img1, img2, uv1, uv2, mask, output: Path, maximum: int = 130, mode: str = "all"
):
    """mode='all': matches SIFT iniciales; 'ransac': inliers verdes/outliers rojos."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.full((max(h1, h2), w1 + w2, 3), 26, dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 : w1 + w2] = img2
    selected = pick_indices(len(uv1), maximum)
    for i in selected:
        p1 = tuple(np.rint(uv1[i]).astype(int))
        p2 = tuple(np.rint(uv2[i]).astype(int) + [w1, 0])
        good = bool(mask[i])
        color = (90, 220, 90) if good else (60, 60, 230)
        if mode == "all":
            color = (200, 200, 200)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 3, color, -1, cv2.LINE_AA)
    title = (
        "MATCHES SIFT (antes de RANSAC)"
        if mode == "all"
        else "RANSAC: VERDE=inlier  ROJO=outlier"
    )
    cv2.rectangle(canvas, (0, 0), (min(w1 + w2 - 1, 970), 40), (13, 13, 13), -1)
    cv2.putText(
        canvas,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    save_cv(output, canvas)


def draw_flow(img1, uv1, uv2, mask, output: Path, maximum: int = 140):
    """Flechas de posición en frame 1 a posición correspondiente en frame 2."""
    canvas = img1.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (min(img1.shape[1] - 1, 850), 72), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, canvas, 0.40, 0, canvas)
    cv2.putText(
        canvas,
        "FLUJO 2D DE FEATURES (inliers)",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Flecha: p(frame 1) -> p(frame 2), en pixeles",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    indices = np.flatnonzero(mask)
    for j in pick_indices(len(indices), maximum):
        i = indices[j]
        p1 = tuple(np.rint(uv1[i]).astype(int))
        p2 = tuple(np.rint(uv2[i]).astype(int))
        color = (255, 190, 50)
        cv2.arrowedLine(canvas, p1, p2, color, 2, cv2.LINE_AA, tipLength=0.23)
        cv2.circle(canvas, p1, 2, (30, 30, 255), -1, cv2.LINE_AA)
    save_cv(output, canvas)


def line_border_endpoints(line: np.ndarray, w: int, h: int):
    a, b, c = line
    candidates = []
    if abs(b) > 1e-12:
        for x in [0.0, float(w - 1)]:
            y = -(a * x + c) / b
            if 0 <= y <= h - 1:
                candidates.append((int(round(x)), int(round(y))))
    if abs(a) > 1e-12:
        for y in [0.0, float(h - 1)]:
            x = -(b * y + c) / a
            if 0 <= x <= w - 1:
                candidates.append((int(round(x)), int(round(y))))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < 2:
        return None
    return candidates[0], candidates[1]


def draw_epipolar(img1, img2, uv1, uv2, mask, E, K, output: Path, maximum: int = 12):
    """Por x1 la línea l2 = F x1; F en píxeles = K^-T E K^-1."""
    left, right = img1.copy(), img2.copy()
    h2, w2 = img2.shape[:2]
    invK = np.linalg.inv(K)
    F = invK.T @ E @ invK
    inlier_ids = np.flatnonzero(mask)
    indices = inlier_ids[pick_indices(len(inlier_ids), maximum, seed=17)]
    hues = np.linspace(0, 179, max(1, len(indices)), endpoint=False).astype(np.uint8)
    hsv = np.column_stack(
        [
            hues,
            np.full(len(indices), 245, np.uint8),
            np.full(len(indices), 255, np.uint8),
        ]
    ).reshape(-1, 1, 3)
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    distances = []
    for idx, col in zip(indices, colors):
        color = tuple(int(v) for v in col)
        p1 = tuple(np.rint(uv1[idx]).astype(int))
        p2 = tuple(np.rint(uv2[idx]).astype(int))
        line = F @ np.array([*uv1[idx], 1.0])
        ends = line_border_endpoints(line, w2, h2)
        if ends is not None:
            cv2.line(right, ends[0], ends[1], color, 2, cv2.LINE_AA)
        cv2.circle(left, p1, 6, color, -1, cv2.LINE_AA)
        cv2.circle(right, p2, 6, color, -1, cv2.LINE_AA)
        dist = abs(line @ np.array([*uv2[idx], 1.0])) / max(
            np.linalg.norm(line[:2]), 1e-12
        )
        distances.append(float(dist))
    canvas = np.hstack((left, right))
    cv2.rectangle(
        canvas, (0, 0), (min(canvas.shape[1] - 1, 1330), 41), (14, 14, 14), -1
    )
    cv2.putText(
        canvas,
        "EPIPOLAR: punto izquierda -> linea derecha (mismo color)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.69,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    save_cv(output, canvas)
    return float(np.median(distances)) if distances else float("nan")


def set_axes_equal(ax):
    """Escala geométrica idéntica de X/Y/Z (evita deformaciones del 3D)."""
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    centers = [np.mean(pair) for pair in (xlim, ylim, zlim)]
    radius = max((hi - lo) / 2 for lo, hi in (xlim, ylim, zlim))
    radius = max(radius, 0.01)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def draw_camera(ax, C: np.ndarray, Q: np.ndarray, scale: float, label: str, color: str):
    """Q transforma un vector de ejes de cámara a ejes de cámara 1."""
    corners = (
        np.array(
            [[-0.75, -0.5, 1.4], [0.75, -0.5, 1.4], [0.75, 0.5, 1.4], [-0.75, 0.5, 1.4]]
        )
        * scale
    )
    corners_world = (Q @ corners.T).T + C
    for p in corners_world:
        ax.plot([C[0], p[0]], [C[1], p[1]], [C[2], p[2]], color=color, lw=1.6)
    closed = np.vstack([corners_world, corners_world[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, lw=1.6)
    ax.scatter(*C, color=color, marker="^", s=90, label=label, depthshade=False)
    for i, axis_color in enumerate(["#d62728", "#2ca02c", "#1f77b4"]):
        end = C + Q[:, i] * scale
        ax.plot(
            [C[0], end[0]], [C[1], end[1]], [C[2], end[2]], color=axis_color, lw=2.3
        )


def draw_cloud_3d(
    X: np.ndarray,
    reliable_mask: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output: Path,
    maximum: int = 800,
    keep_open: bool = False,
):
    C1 = np.zeros(3)
    C2 = -R.T @ t
    cloud = X[reliable_mask]
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    cloud = cloud[pick_indices(len(cloud), maximum)]
    # Extremos muy lejanos se excluyen SOLO de la VISTA, no del cálculo de pose.
    if len(cloud) >= 10:
        low = np.percentile(cloud, 3, axis=0)
        high = np.percentile(cloud, 97, axis=0)
        cloud = cloud[np.all((cloud >= low) & (cloud <= high), axis=1)]
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection="3d")
    if len(cloud):
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            cloud[:, 2],
            c=cloud[:, 2],
            cmap="viridis",
            alpha=0.73,
            s=10,
            label=f"Puntos triangulados filtrados ({len(cloud)})",
        )
    spread = (
        (np.max(cloud, axis=0) - np.min(cloud, axis=0)).max() if len(cloud) else 2.0
    )
    camera_scale = float(np.clip(spread * 0.06, 0.15, 1.5))
    draw_camera(ax, C1, np.eye(3), camera_scale, "Cámara 1 (origen)", "#d62728")
    draw_camera(ax, C2, R.T, camera_scale, "Cámara 2 (pose relativa)", "#ff7f0e")
    ax.plot(
        [0, C2[0]],
        [0, C2[1]],
        [0, C2[2]],
        "--",
        c="black",
        lw=1.2,
        label="Desplazamiento (longitud arbitraria)",
    )
    ax.set_xlabel("X cámara 1 [escala arbitraria]")
    ax.set_ylabel("Y cámara 1 [escala arbitraria]")
    ax.set_zlabel("Z cámara 1 [escala arbitraria]")
    ax.set_title(
        "Triangulación y poses en el sistema de cámara 1\n"
        "Rojo/verde/azul = ejes locales X/Y/Z; +Z apunta hacia delante"
    )
    ax.view_init(elev=18, azim=-65)
    ax.legend(loc="upper left", fontsize=8)
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    if not keep_open:
        plt.close(fig)
    return len(cloud)


def draw_overview(output_dir: Path):
    names = [
        "01_matches.png",
        "02_ransac.png",
        "03_flujo_2d.png",
        "04_epipolar.png",
        "05_reconstruccion_3d.png",
    ]
    labels = [
        "1. Matches SIFT",
        "2. RANSAC",
        "3. Movimiento aparente",
        "4. Restricción epipolar",
        "5. Dos cámaras + mapa 3D",
    ]
    fig = plt.figure(figsize=(17, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.4])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),
    ]
    for ax, name, label in zip(axes, names, labels):
        img = plt.imread(output_dir / name)
        ax.imshow(img)
        ax.set_title(label, fontsize=13, weight="bold")
        ax.axis("off")
    fig.suptitle("Visual Odometry monocular — dos vistas", fontsize=19, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "00_resumen.png", dpi=135, bbox_inches="tight")
    plt.close(fig)


def estimate_pair(
    img1,
    img2,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    iterations=2500,
    min_parallax_deg=0.2,
):
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        raise ValueError("Ambos fotogramas deben tener iguales dimensiones")
    fx = float(fx if fx is not None else 0.85 * w)
    fy = float(fy if fy is not None else fx)
    cx = float(cx if cx is not None else (w - 1) / 2)
    cy = float(cy if cy is not None else (h - 1) / 2)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    sift = cv2.SIFT_create(nfeatures=2500)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        raise RuntimeError("SIFT no detectó suficientes features en ambos fotogramas")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    matches = [
        m
        for pair in knn
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.72 * n.distance
    ]
    if len(matches) < 20:
        raise RuntimeError(f"Solo {len(matches)} matches. Prueba otro par de frames")
    uv1 = np.array([k1[m.queryIdx].pt for m in matches], dtype=float)
    uv2 = np.array([k2[m.trainIdx].pt for m in matches], dtype=float)
    x1 = normalized_image_points(uv1, K)
    x2 = normalized_image_points(uv2, K)
    E, mask = ransac_essential(
        x1, x2, threshold=1.5 / ((fx + fy) / 2), iterations=iterations
    )
    R, t, positive_sample, sample_count = decompose_pose(E, x1[mask], x2[mask])
    C2 = -R.T @ t
    X = triangulate_all(R, t, x1[mask], x2[mask])
    tri = triangulation_diagnostics(
        K, R, t, X, x1[mask], x2[mask], uv1[mask], uv2[mask], min_parallax_deg
    )
    return {
        "K": K,
        "uv1": uv1,
        "uv2": uv2,
        "x1": x1,
        "x2": x2,
        "matches": len(matches),
        "inliers": int(mask.sum()),
        "mask": mask,
        "E": E,
        "R_2_from_1": R,
        "t_2_from_1": t,
        "center_2_in_1": C2,
        "rpy_deg_camera_2_in_1": euler_zyx_deg(R.T),
        "positive_depth_sample": positive_sample,
        "sample_count": sample_count,
        "X": X,
        "tri": tri,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", help="MP4 del Tello")
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--second", type=int, default=4)
    parser.add_argument("--fx", type=float, default=None, help="Focal X calibrada (px)")
    parser.add_argument("--fy", type=float, default=None, help="Focal Y calibrada (px)")
    parser.add_argument("--cx", type=float, default=None, help="Centro óptico X (px)")
    parser.add_argument("--cy", type=float, default=None, help="Centro óptico Y (px)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultados_vo"),
        help="Carpeta en la que guardar figuras e informe",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abrir resumen y figura 3D giratoria en matplotlib al terminar",
    )
    parser.add_argument("--ransac-iterations", type=int, default=2500)
    parser.add_argument(
        "--min-parallax",
        type=float,
        default=0.2,
        help="Paralaje angular mínimo de la nube visualizada (grados)",
    )
    args = parser.parse_args()
    if args.first < 0 or args.second <= args.first:
        parser.error("Es necesario 0 <= --first < --second")
    if args.min_parallax < 0:
        parser.error("--min-parallax debe ser >= 0")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        parser.error(f"No puedo abrir el vídeo: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        img1 = read_frame(cap, args.first)
        img2 = read_frame(cap, args.second)
    finally:
        cap.release()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    print("Estimando correspondencias, E, R, t y triangulación...")
    result = estimate_pair(
        img1,
        img2,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        args.ransac_iterations,
        args.min_parallax,
    )
    uv1, uv2, mask = result["uv1"], result["uv2"], result["mask"]
    K, E, R, t, C2 = (
        result[k] for k in ["K", "E", "R_2_from_1", "t_2_from_1", "center_2_in_1"]
    )
    X, tri = result["X"], result["tri"]
    print("Generando visualizaciones...")
    save_cv(output / "frame_1.png", img1)
    save_cv(output / "frame_2.png", img2)
    draw_matches(img1, img2, uv1, uv2, mask, output / "01_matches.png", mode="all")
    draw_matches(img1, img2, uv1, uv2, mask, output / "02_ransac.png", mode="ransac")
    draw_flow(img1, uv1, uv2, mask, output / "03_flujo_2d.png")
    epipolar_error = draw_epipolar(
        img1, img2, uv1, uv2, mask, E, K, output / "04_epipolar.png"
    )
    cloud_visible = draw_cloud_3d(
        X,
        tri["reliable_mask"],
        R,
        t,
        output / "05_reconstruccion_3d.png",
        keep_open=args.show,
    )
    draw_overview(output)
    valid = tri["positive_mask"]
    parallaxes = tri["parallax_deg"][valid]
    errors = tri["reprojection_px"][valid]
    report = {
        "video": str(args.video),
        "frame_1": args.first,
        "frame_2": args.second,
        "seconds_between_frames": (
            (args.second - args.first) / fps if fps and fps > 0 else None
        ),
        "intrinsics_approximate": args.fx is None or args.fy is None,
        "K": K.tolist(),
        "initial_matches": result["matches"],
        "inliers_ransac": result["inliers"],
        "inlier_ratio": result["inliers"] / result["matches"],
        "E": E.tolist(),
        "R_2_from_1": R.tolist(),
        "t_2_from_1_unit": t.tolist(),
        "camera_2_center_in_camera_1_units": C2.tolist(),
        "relative_camera_2_rpy_degrees": [
            float(v) for v in result["rpy_deg_camera_2_in_1"]
        ],
        "positive_depth_in_sample": result["positive_depth_sample"],
        "sample_depth_checked": result["sample_count"],
        "positive_depth_in_all_inliers": int(valid.sum()),
        "triangulated_total": len(X),
        "points_after_reprojection_and_parallax_filter": int(
            tri["reliable_mask"].sum()
        ),
        "plotted_3d_points": cloud_visible,
        "median_parallax_deg_positive": (
            float(np.median(parallaxes)) if len(parallaxes) else None
        ),
        "median_reprojection_error_px_positive": (
            float(np.median(errors)) if len(errors) else None
        ),
        "median_epipolar_distance_sample_px": epipolar_error,
        "min_parallax_for_visualization_deg": args.min_parallax,
        "warning": (
            "La EIS y una K aproximada pueden afectar el resultado. "
            "Traslacion solo conocida hasta escala y pose potencialmente degenerada."
        ),
    }
    (output / "informe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Exportar puntos 3D junto a mediciones para inspeccionarlos después.
    csv_table = np.column_stack(
        [
            X,
            tri["positive_mask"].astype(int),
            tri["reliable_mask"].astype(int),
            tri["parallax_deg"],
            tri["reprojection_px"],
        ]
    )
    np.savetxt(
        output / "puntos_3d.csv",
        csv_table,
        delimiter=",",
        comments="",
        header="X,Y,Z,profundidad_positiva,filtrado_ok,paralaje_grados,error_reproyeccion_px",
        fmt="%.7g",
    )
    np.set_printoptions(precision=4, suppress=True)
    print("\nMatriz K (aproximada salvo que pases calibración):\n", K)
    print(f"Matches {result['matches']} | Inliers RANSAC {result['inliers']} ")
    print("Esencial E:\n", E)
    print("Rotación R (coordenadas cámara 1 -> cámara 2):\n", R)
    print("t (transformación 1 -> 2, norma 1):", t)
    print("Centro de cámara 2 expresado en cámara 1:", C2)
    print(
        "Orientación cámara 2 en cámara 1: roll/pitch/yaw grados:",
        result["rpy_deg_camera_2_in_1"],
    )
    print(
        f"Profundidad positiva: {result['positive_depth_sample']}/{result['sample_count']} "
        f"en test; {int(valid.sum())}/{len(X)} entre todos los inliers"
    )
    print(
        f"Puntos 3D tras filtros de reproyección/paralaje: {int(tri['reliable_mask'].sum())}"
    )
    print(f"Mediana paralaje [grados]: {report['median_parallax_deg_positive']}")
    print(
        f"Mediana error reproyección [px]: {report['median_reprojection_error_px_positive']}"
    )
    if result["positive_depth_sample"] < 0.7 * result["sample_count"]:
        print(
            "AVISO: quiralidad ambigua: la dirección de traslación y la nube pueden ser INESTABLES."
        )
    if (
        report["median_parallax_deg_positive"] is not None
        and report["median_parallax_deg_positive"] < 0.5
    ):
        print(
            "AVISO: paralaje pequeño; prueba fotogramas más separados y con TRASLACIÓN."
        )
    print(f"\nArchivos guardados en: {output.resolve()}")
    print(
        "Abre primero 00_resumen.png y después 04_epipolar.png / 05_reconstruccion_3d.png"
    )
    print(
        "ATENCIÓN: la longitud de t = 1 no implica 1 metro. EIS puede alterar la pose física."
    )
    if args.show:
        print(
            "Con --show, arrastra con el ratón la ventana 3D para girar la reconstrucción."
        )
        summary = plt.imread(output / "00_resumen.png")
        plt.figure(figsize=(15, 10))
        plt.imshow(summary)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
