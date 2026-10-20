"""Visual Odometry monocular con visualizaciones grandes en Plotly.

Este script reutiliza la geometría del archivo VO_2_frames_visual.py
(correspondencias, matriz esencial, R/t, triangulación) y cambia la
presentación visual: genera un informe HTML grande e interactivo en lugar de
un resumen pequeño con matplotlib.

Dependencias:
    pip install numpy opencv-python plotly

Ejemplo:
    python VO_2_frames_plotly.py video.mp4 --first 1 --second 11
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "VO_2_frames_visual.py"


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"No encuentro el script base: {BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("vo_visual_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def camera_geometry(C: np.ndarray, Q: np.ndarray, scale: float) -> np.ndarray:
    """Vértices usados al dibujar la cámara, incluidos sus ejes locales.

    Los rangos del gráfico DEBEN incluir estos puntos (y no solo C1/C2).
    """
    corners = (
        np.array(
            [
                [-0.75, -0.5, 1.4],
                [0.75, -0.5, 1.4],
                [0.75, 0.5, 1.4],
                [-0.75, 0.5, 1.4],
            ]
        )
        * scale
    )
    points = (Q @ corners.T).T + C
    axes = C + scale * Q.T
    return np.vstack([C.reshape(1, 3), points, axes])


def camera_wireframe_traces(
    C: np.ndarray, Q: np.ndarray, scale: float, label: str, color_body: str
) -> List[go.Scatter3d]:
    """Q transforma ejes de cámara local a ejes de cámara 1."""
    points = camera_geometry(C, Q, scale)
    corners_world = points[1:5]
    traces: List[go.Scatter3d] = []
    traces.append(
        go.Scatter3d(
            x=[C[0]],
            y=[C[1]],
            z=[C[2]],
            mode="markers+text",
            marker=dict(size=6, color=color_body, symbol="diamond"),
            text=[label],
            textposition="top center",
            textfont=dict(size=11),
            name=label,
            showlegend=True,
        )
    )
    for p in corners_world:
        traces.append(
            go.Scatter3d(
                x=[C[0], p[0]],
                y=[C[1], p[1]],
                z=[C[2], p[2]],
                mode="lines",
                line=dict(color=color_body, width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    closed = np.vstack([corners_world, corners_world[0]])
    traces.append(
        go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=closed[:, 2],
            mode="lines",
            line=dict(color=color_body, width=5),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    for axis_name, axis_color, i in [
        ("X", "#d62728", 0),
        ("Y", "#2ca02c", 1),
        ("Z", "#1f77b4", 2),
    ]:
        end = C + Q[:, i] * scale
        traces.append(
            go.Scatter3d(
                x=[C[0], end[0]],
                y=[C[1], end[1]],
                z=[C[2], end[2]],
                mode="lines",
                line=dict(color=axis_color, width=7),
                name=f"{label}: eje {axis_name}",
                showlegend=False,
                hovertemplate=f"{label}<br>Eje {axis_name}<extra></extra>",
            )
        )
    return traces


def equal_axis_ranges(points: np.ndarray, padding: float = 0.25):
    """Cubo centrado con un margen adicional alrededor de toda la geometría."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        pts = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = (mins + maxs) / 2.0
    half_side = max(float(np.max(maxs - mins)) / 2.0, 0.3) * (1.0 + padding)
    return dict(
        zip(
            ("x", "y", "z"),
            [[float(c - half_side), float(c + half_side)] for c in center],
        )
    )


def visible_cloud(X: np.ndarray, max_points: int = 1200) -> np.ndarray:
    """Solo limita puntos extremos para la VISTA, sin cambiar las estimaciones."""
    cloud = X[np.isfinite(X).all(axis=1)]
    if len(cloud) > max_points:
        cloud = cloud[base.pick_indices(len(cloud), max_points)]
    if len(cloud) >= 20:
        low = np.percentile(cloud, 2, axis=0)
        high = np.percentile(cloud, 98, axis=0)
        trimmed = cloud[np.all((cloud >= low) & (cloud <= high), axis=1)]
        if len(trimmed):
            cloud = trimmed
    return cloud


def build_3d_figure(
    X: np.ndarray,
    reliable_mask: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    maximum: int = 1200,
    positive_mask: np.ndarray | None = None,
) -> go.Figure:
    C1 = np.zeros(3)
    C2 = -R.T @ t
    valid = np.asarray(reliable_mask, dtype=bool)
    positive = valid if positive_mask is None else np.asarray(positive_mask, dtype=bool)
    if len(valid) != len(X) or len(positive) != len(X):
        raise ValueError("Las máscaras 3D deben tener la misma longitud que X")

    reliable = visible_cloud(X[valid], maximum)
    tentative = visible_cloud(X[positive & ~valid], maximum)
    # Base visual estable: el tamaño de las cámaras depende de la separación
    # entre ellas, no de posibles puntos triangulados muy lejanos.
    baseline = float(np.linalg.norm(C2 - C1))
    camera_scale = max(0.18, baseline * 0.22)
    geom = np.vstack(
        [
            camera_geometry(C1, np.eye(3), camera_scale),
            camera_geometry(C2, R.T, camera_scale),
        ]
    )

    # La vista inicial se centra en las cámaras, incluyendo TODO el volumen
    # de sus frustums y los ejes coloreados. No dependemos de la nube.
    camera_ranges = equal_axis_ranges(geom, padding=0.40)
    # La segunda vista muestra la nube de profundidad positiva (recortando
    # solo puntos extremísimos para que un outlier no arruine el encuadre).
    point_arrays = [a for a in [reliable, tentative] if len(a)]
    all_geometry = np.vstack([geom] + point_arrays)
    map_ranges = equal_axis_ranges(all_geometry, padding=0.20)

    fig = go.Figure()
    if len(reliable):
        fig.add_trace(
            go.Scatter3d(
                x=reliable[:, 0],
                y=reliable[:, 1],
                z=reliable[:, 2],
                mode="markers",
                marker=dict(
                    size=3.5,
                    color=reliable[:, 2],
                    colorscale="Viridis",
                    opacity=0.85,
                    colorbar=dict(title="Profundidad Z", thickness=12),
                ),
                name=f"Puntos fiables ({len(reliable)})",
                hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
            )
        )
    if len(tentative):
        fig.add_trace(
            go.Scatter3d(
                x=tentative[:, 0],
                y=tentative[:, 1],
                z=tentative[:, 2],
                mode="markers",
                marker=dict(size=2.5, color="#a3a3a3", opacity=0.45),
                visible="legendonly" if len(reliable) else True,
                name=f"Puntos descartados por filtros ({len(tentative)})",
                hovertemplate="NO VALIDADO<br>X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
            )
        )

    for tr in camera_wireframe_traces(
        C1, np.eye(3), camera_scale, "Cámara 1", "#d62728"
    ):
        fig.add_trace(tr)
    for tr in camera_wireframe_traces(C2, R.T, camera_scale, "Cámara 2", "#ff7f0e"):
        fig.add_trace(tr)
    fig.add_trace(
        go.Scatter3d(
            x=[C1[0], C2[0]],
            y=[C1[1], C2[1]],
            z=[C1[2], C2[2]],
            mode="lines",
            line=dict(color="#334155", width=4, dash="dash"),
            name="Desplazamiento (escala arbitraria)",
            hovertemplate="Separación arbitraria entre cámaras<extra></extra>",
        )
    )

    # Los botones permiten desacoplar el encuadre de cámaras de los puntos,
    # que con bajo paralaje pueden quedar a distancias arbitrariamente grandes.
    def relayout_for(ranges):
        return {
            "scene.xaxis.range": ranges["x"],
            "scene.yaxis.range": ranges["y"],
            "scene.zaxis.range": ranges["z"],
            "scene.camera": dict(
                eye=dict(x=1.55, y=-1.6, z=1.2), projection=dict(type="orthographic")
            ),
        }

    subtitle = (
        "Puntos 3D fiables: "
        + str(len(reliable))
        + " | Candidatos descartados: "
        + str(len(tentative))
        + " | +Z = dirección de visión | unidades arbitrarias"
    )
    fig.update_layout(
        title=dict(
            text="Dos cámaras y triangulación 3D<br><sup>" + subtitle + "</sup>", x=0.02
        ),
        scene=dict(
            domain=dict(x=[0.03, 0.97], y=[0.02, 0.98]),
            xaxis=dict(title="X", range=camera_ranges["x"], showbackground=True),
            yaxis=dict(title="Y", range=camera_ranges["y"], showbackground=True),
            zaxis=dict(title="Z", range=camera_ranges["z"], showbackground=True),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.55, y=-1.6, z=1.2), projection=dict(type="orthographic")
            ),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.02,
                y=1.07,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Encuadrar cámaras",
                        method="relayout",
                        args=[relayout_for(camera_ranges)],
                    ),
                    dict(
                        label="Encuadrar nube + cámaras",
                        method="relayout",
                        args=[relayout_for(map_ranges)],
                    ),
                ],
            )
        ],
        annotations=[
            dict(
                x=0.5,
                y=-0.07,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=(
                    "Arrastra para girar · Rueda para zoom · Leyenda para mostrar/ocultar puntos"
                    " · Ejes en escala arbitraria"
                ),
                font=dict(size=12, color="#64748b"),
            )
        ],
        template="plotly_white",
        height=900,
        legend=dict(
            orientation="h",
            x=0.02,
            y=1.0,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
        ),
        margin=dict(l=40, r=70, t=165, b=100),
        uirevision="vo3d",
    )
    return fig


def build_image_figure(image_path: Path, title: str) -> go.Figure:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No puedo leer la imagen: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # Altura limitada pero grande; si quieres más, sube el tope.
    fig_height = int(min(max(h, 520), 980))
    fig = go.Figure(go.Image(z=img))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, scaleanchor="x")
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=min(max(w, 900), 1650),
        height=fig_height,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def write_html_dashboard(output_dir: Path, fig_3d: go.Figure):
    sections = [
        ("01_matches.png", "1. Matches SIFT — correspondencias iniciales"),
        ("02_ransac.png", "2. RANSAC — inliers vs outliers"),
        ("03_flujo_2d.png", "3. Flujo 2D — movimiento aparente en la imagen"),
        (
            "04_epipolar.png",
            "4. Restricción epipolar — punto izquierda → línea derecha",
        ),
    ]

    html_parts = [
        '<html><head><meta charset="utf-8"><title>VO monocular — informe Plotly</title></head>',
        '<body style="font-family:Arial,sans-serif;max-width:1700px;margin:0 auto;padding:18px;background:#fafafa;">',
        "<h1>Visual Odometry monocular — informe interactivo</h1>",
        "<p>Los gráficos 2D permiten ampliar con zoom. En la vista 3D puedes encuadrar cámaras o nube sin perder la geometría.</p>",
    ]

    include_js = True
    for filename, title in sections:
        fig = build_image_figure(output_dir / filename, title)
        html_parts.append(
            pio.to_html(fig, full_html=False, include_plotlyjs=include_js)
        )
        include_js = False

    html_parts.append(pio.to_html(fig_3d, full_html=False, include_plotlyjs=False))
    html_parts.append("</body></html>")

    (output_dir / "00_resumen_plotly.html").write_text(
        "".join(html_parts), encoding="utf-8"
    )
    # Guardar el 3D también por separado
    (output_dir / "05_reconstruccion_3d_plotly.html").write_text(
        pio.to_html(fig_3d, full_html=True, include_plotlyjs=True), encoding="utf-8"
    )


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
        default=Path("resultados_vo_plotly"),
        help="Carpeta en la que guardar figuras e informe",
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
        img1 = base.read_frame(cap, args.first)
        img2 = base.read_frame(cap, args.second)
    finally:
        cap.release()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    print("Estimando correspondencias, E, R, t y triangulación...")
    result = base.estimate_pair(
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

    print("Generando PNGs base y HTML de Plotly...")
    base.save_cv(output / "frame_1.png", img1)
    base.save_cv(output / "frame_2.png", img2)
    base.draw_matches(img1, img2, uv1, uv2, mask, output / "01_matches.png", mode="all")
    base.draw_matches(
        img1, img2, uv1, uv2, mask, output / "02_ransac.png", mode="ransac"
    )
    base.draw_flow(img1, uv1, uv2, mask, output / "03_flujo_2d.png")
    epipolar_error = base.draw_epipolar(
        img1, img2, uv1, uv2, mask, E, K, output / "04_epipolar.png"
    )
    fig3d = build_3d_figure(
        X, tri["reliable_mask"], R, t, positive_mask=tri["positive_mask"]
    )
    write_html_dashboard(output, fig3d)

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
            "Traslación solo conocida hasta escala y pose potencialmente degenerada."
        ),
    }
    (output / "informe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
    print(f"Matches {result['matches']} | Inliers RANSAC {result['inliers']}")
    print("Esencial E:\n", E)
    print("Rotación R (coordenadas cámara 1 -> cámara 2):\n", R)
    print("t (transformación 1 -> 2, norma 1):", t)
    print("Centro de cámara 2 expresado en cámara 1:", C2)
    print(
        "Orientación cámara 2 en cámara 1: roll/pitch/yaw grados:",
        result["rpy_deg_camera_2_in_1"],
    )
    print(
        f"Profundidad positiva: {result['positive_depth_sample']}/{result['sample_count']} en test; "
        f"{int(valid.sum())}/{len(X)} entre todos los inliers"
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
    print("Abre en el navegador: 00_resumen_plotly.html")
    print("También tienes: 05_reconstruccion_3d_plotly.html")
    print(
        "ATENCIÓN: la longitud de t = 1 no implica 1 metro. EIS puede alterar la pose física."
    )


if __name__ == "__main__":
    main()
