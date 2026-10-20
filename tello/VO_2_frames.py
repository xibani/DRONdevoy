"""Estimación de pose relativa monocular usando dos fotogramas.

Dependencias: pip install numpy opencv-python
Ejemplo: python vo_dos_frames.py grabacion.mp4 --first 50 --second 75

Los puntos 2D se extraen con SIFT (OpenCV). La geometría se implementa aquí:
normalización de Hartley, algoritmo de 8 puntos, RANSAC, descomposición
esencial, selección por quiralidad y triangulación lineal.
"""

import argparse
import numpy as np
import cv2


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"No se puede leer el fotograma {idx}")
    return frame


def homogeneous(x):
    return np.column_stack([x, np.ones(len(x))])


def hartley_normalization(x):
    center = np.mean(x, axis=0)
    mean_radius = np.mean(np.linalg.norm(x - center, axis=1))
    if mean_radius < 1e-12:
        raise ValueError("Puntos degenerados")
    s = np.sqrt(2.0) / mean_radius
    T = np.array([[s, 0, -s * center[0]], [0, s, -s * center[1]], [0, 0, 1.0]])
    xn = (T @ homogeneous(x).T).T
    return xn[:, :2] / xn[:, 2:3], T


def essential_eight_points(x1, x2):
    """x1, x2: Nx2 coordenadas de cámara ya normalizadas con K^-1."""
    p1, T1 = hartley_normalization(x1)
    p2, T2 = hartley_normalization(x2)
    a, b = p1[:, 0], p1[:, 1]
    c, d = p2[:, 0], p2[:, 1]
    A = np.column_stack([c * a, c * b, c, d * a, d * b, d, a, b, np.ones(len(a))])
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0.0]) @ Vt  # rango 2
    E = T2.T @ F @ T1
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt  # singularidades (1,1,0)
    return E / np.linalg.norm(E)


def sampson_error(E, x1, x2):
    a, b = homogeneous(x1), homogeneous(x2)
    Ea = (E @ a.T).T
    Etb = (E.T @ b.T).T
    e = np.sum(b * Ea, axis=1)
    return e**2 / (
        Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2 + 1e-15
    )


def ransac_essential(x1, x2, threshold=0.002, iterations=2500):
    rng = np.random.default_rng(7)
    best_mask = None
    best_count = 0
    for _ in range(iterations):
        ids = rng.choice(len(x1), size=8, replace=False)
        try:
            E = essential_eight_points(x1[ids], x2[ids])
        except (np.linalg.LinAlgError, ValueError):
            continue
        mask = sampson_error(E, x1, x2) < threshold**2
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < 12:
        raise RuntimeError("Correspondencias geométricas insuficientes")
    E = essential_eight_points(x1[best_mask], x2[best_mask])
    refined = sampson_error(E, x1, x2) < threshold**2
    if refined.sum() >= 12:
        E = essential_eight_points(x1[refined], x2[refined])
        best_mask = refined
    return E, best_mask


def triangulate_point(P1, P2, x1, x2):
    A = np.array(
        [
            x1[0] * P1[2] - P1[0],
            x1[1] * P1[2] - P1[1],
            x2[0] * P2[2] - P2[0],
            x2[1] * P2[2] - P2[1],
        ]
    )
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-10:
        return None
    return Xh[:3] / Xh[3]


def decompose_pose(E, x1, x2):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotations = [U @ W @ Vt, U @ W.T @ Vt]
    direction = U[:, 2]
    P1 = np.column_stack([np.eye(3), np.zeros(3)])
    best = (-1, None, None)
    sample = np.linspace(0, len(x1) - 1, min(100, len(x1)), dtype=int)
    for R in rotations:
        for t in (direction, -direction):
            P2 = np.column_stack([R, t])
            positive = 0
            for idx in sample:
                X = triangulate_point(P1, P2, x1[idx], x2[idx])
                if X is not None and X[2] > 0 and (R @ X + t)[2] > 0:
                    positive += 1
            if positive > best[0]:
                best = (positive, R, t.copy())
    return best[1], best[2], best[0], len(sample)


def euler_zyx_deg(R):
    """Euler ZYX en grados, rotación de la cámara respecto al fotograma inicial."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


def estimate_pair(img1, img2, fx=None, fy=None, cx=None, cy=None):
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        raise ValueError("Ambos fotogramas deben tener el mismo tamaño")
    # Solo un K orientativo: sustituir por calibración real de la señal de vídeo.
    fx = float(fx if fx is not None else 0.85 * w)
    fy = float(fy if fy is not None else fx)
    cx = float(cx if cx is not None else (w - 1) / 2)
    cy = float(cy if cy is not None else (h - 1) / 2)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    sift = cv2.SIFT_create(nfeatures=2500)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        raise RuntimeError("No hay puntos de interés suficientes")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    matches = [
        m
        for pair in knn
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.72 * n.distance
    ]
    if len(matches) < 20:
        raise RuntimeError(f"Solo {len(matches)} correspondencias; cambia el intervalo")
    uv1 = np.array([k1[m.queryIdx].pt for m in matches])
    uv2 = np.array([k2[m.trainIdx].pt for m in matches])
    invK = np.linalg.inv(K)
    x1h = (invK @ homogeneous(uv1).T).T
    x2h = (invK @ homogeneous(uv2).T).T
    x1, x2 = x1h[:, :2] / x1h[:, 2:3], x2h[:, :2] / x2h[:, 2:3]
    E, mask = ransac_essential(x1, x2, threshold=1.5 / ((fx + fy) / 2))
    R, t, positive, tested = decompose_pose(E, x1[mask], x2[mask])
    C2 = -R.T @ t  # centro de cámara 2 en el marco de cámara 1
    roll, pitch, yaw = euler_zyx_deg(R.T)
    return {
        "K": K,
        "matches": len(matches),
        "inliers": int(mask.sum()),
        "E": E,
        "R_2_from_1": R,
        "t_2_from_1": t,
        "center_2_in_1": C2,
        "rpy_deg_camera_2_in_1": (roll, pitch, yaw),
        "positive_depth": positive,
        "tested_depth": tested,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Ruta al vídeo MP4 grabado por Tello")
    parser.add_argument("--first", type=int, default=50)
    parser.add_argument("--second", type=int, default=75)
    parser.add_argument(
        "--fx", type=float, default=None, help="Focal horizontal calibrada en píxeles"
    )
    parser.add_argument(
        "--fy", type=float, default=None, help="Focal vertical calibrada en píxeles"
    )
    args = parser.parse_args()
    if args.second <= args.first:
        parser.error("--second debe ser mayor que --first")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        parser.error("No se pudo abrir el vídeo")
    try:
        im1 = read_frame(cap, args.first)
        im2 = read_frame(cap, args.second)
    finally:
        cap.release()
    result = estimate_pair(im1, im2, fx=args.fx, fy=args.fy)
    np.set_printoptions(precision=4, suppress=True)
    print("Matriz K aproximada:\n", result["K"])
    print("Matches:", result["matches"], "Inliers:", result["inliers"])
    print("Matriz esencial E:\n", result["E"])
    print("Rotación R (de cámara 1 a cámara 2):\n", result["R_2_from_1"])
    print(
        "Traslación t de la transformación 1->2 (norma unitaria):", result["t_2_from_1"]
    )
    print(
        "Centro cámara 2 en coordenadas cámara 1 (escala arbitraria):",
        result["center_2_in_1"],
    )
    print(
        "Orientación relativa roll/pitch/yaw [grados]:", result["rpy_deg_camera_2_in_1"]
    )
    print(
        "Triangulaciones con profundidad positiva:",
        f'{result["positive_depth"]}/{result["tested_depth"]}',
    )
    print(
        "ATENCIÓN: escala métrica no observable; orientación afectada por posible EIS."
    )


if __name__ == "__main__":
    main()
