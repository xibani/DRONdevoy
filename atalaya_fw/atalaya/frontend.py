"""
atalaya.frontend — tracker KLT + medidas de pose relativa entre keyframes.

Portado literalmente de las Fases 4/5 del curso, parametrizado por la cámara
(sin globales). Principios ya validados que este módulo respeta:

* Se rastrea sobre la imagen CRUDA; solo se indistorsionan los keypoints
  (enfoque B de la Fase 2).
* La puerta de admisión de keyframe usa PARALAJE DEROTADO, no flujo bruto:
  una rotación pura produce mucho flujo con paralaje cero (trampa documentada
  en la Fase 6B).
* Las medidas de bajo paralaje se emiten MARCADAS, no se descartan: el gating
  χ² del filtro decide.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from .geometria import Log, inv_T, make_T
from .sensores import CamaraPinhole

METODO_RANSAC = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC


@dataclass
class ConfigFrontend:
    max_feats: int = 250
    min_feats: int = 150
    calidad: float = 0.01
    dist_min: int = 18
    ventana: tuple = (21, 21)
    niveles: int = 3
    umbral_fb: float = 1.0
    kf_min: int = 2
    kf_max: int = 8
    paralaje_min: float = 2.0     # px, DEROTADO
    min_comunes: int = 40
    umbral_ransac_px: float = 1.0


class RastreadorKLT:
    """KLT piramidal con IDs persistentes, forward-backward y redetección
    enmascarada. Idéntico al de la Fase 4/5."""

    def __init__(self, cfg: ConfigFrontend = None):
        c = cfg or ConfigFrontend()
        self.max_feats, self.min_feats = c.max_feats, c.min_feats
        self.calidad, self.dist_min = c.calidad, c.dist_min
        self.umbral_fb = c.umbral_fb
        self.lk = dict(winSize=tuple(c.ventana), maxLevel=c.niveles,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.img_prev = None
        self.pts = np.zeros((0, 2), np.float32)
        self.ids = np.zeros((0,), np.int64)
        self._sig_id = 0
        self.tracks = defaultdict(list)

    def _detectar(self, img):
        n_nuevas = self.max_feats - len(self.pts)
        if n_nuevas <= 0:
            return
        mascara = np.full(img.shape, 255, np.uint8)
        for p in self.pts:
            cv2.circle(mascara, (int(round(float(p[0]))), int(round(float(p[1])))),
                       self.dist_min, 0, -1)
        nuevas = cv2.goodFeaturesToTrack(img, n_nuevas, self.calidad, self.dist_min,
                                         mask=mascara, blockSize=7)
        if nuevas is None:
            return
        nuevas = nuevas.reshape(-1, 2).astype(np.float32)
        ids = np.arange(self._sig_id, self._sig_id + len(nuevas), dtype=np.int64)
        self._sig_id += len(nuevas)
        self.pts = np.vstack([self.pts, nuevas]).astype(np.float32)
        self.ids = np.concatenate([self.ids, ids])

    def _anotar(self, k):
        for i, p in zip(self.ids, self.pts):
            self.tracks[int(i)].append((k, float(p[0]), float(p[1])))

    def rastrear(self, img, k):
        if self.img_prev is None or len(self.pts) == 0:
            self.img_prev = img
            self._detectar(img)
            self._anotar(k)
            return self.ids.copy(), None, self.pts.copy()

        pts_prev = self.pts.reshape(-1, 1, 2)
        cur, st, _ = cv2.calcOpticalFlowPyrLK(self.img_prev, img, pts_prev, None, **self.lk)
        atras, st2, _ = cv2.calcOpticalFlowPyrLK(img, self.img_prev, cur, None, **self.lk)
        fb = np.linalg.norm(pts_prev - atras, axis=2).ravel()

        h, w = img.shape
        c = cur.reshape(-1, 2)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.umbral_fb)
        ok &= (c[:, 0] > 3) & (c[:, 0] < w - 4) & (c[:, 1] > 3) & (c[:, 1] < h - 4)

        ids_ok = self.ids[ok]
        prev_ok = self.pts[ok].astype(np.float32)
        cur_ok = c[ok].astype(np.float32)

        self.pts, self.ids, self.img_prev = cur_ok, ids_ok, img
        if len(self.pts) < self.min_feats:
            self._detectar(img)
        self._anotar(k)
        return ids_ok, prev_ok, cur_ok


def pose_relativa(n0, n1, f_media, umbral_px=1.0, prob=0.999):
    """n0, n1 normalizadas ideales -> (T_c1_c0, mask inliers) o (None, None)."""
    if len(n0) < 8:
        return None, None
    n0 = np.ascontiguousarray(n0, dtype=np.float64)
    n1 = np.ascontiguousarray(n1, dtype=np.float64)
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), method=METODO_RANSAC,
                                   prob=prob, threshold=umbral_px / f_media)
    if E is None or E.shape[0] < 3:
        return None, None
    n_in, R, t, mask_pose = cv2.recoverPose(E[:3], n0, n1, np.eye(3), mask=mask.copy())
    if n_in < 10:
        return None, None
    return make_T(R, t.ravel()), (mask_pose.ravel() > 0)


def paralaje_derotada(n0, n1, R, f_media):
    """Residuo en píxeles tras compensar la rotación R (= R_c1_c0)."""
    d0 = np.hstack([n0, np.ones((len(n0), 1))])
    v = d0 @ R.T
    pred = v[:, :2] / v[:, 2:3]
    return np.linalg.norm(n1 - pred, axis=1) * f_media


def construir_medidas(seq, k0, k1, cfg: ConfigFrontend = None, verbose=True,
                      con_gt=True):
    """Recorre el vídeo una vez y devuelve la lista de medidas visuales.

    Cada medida: dict con k_kf, k, T_c0c1 (rotación + dirección UNITARIA),
    paralaje, n_comunes, inliers, bajo_paralaje, y si hay GT: s_gt, err_rot,
    err_dir (SOLO diagnóstico; jamás entran al filtro).
    """
    c = cfg or ConfigFrontend()
    cam: CamaraPinhole = seq.camara
    fm = cam.f_media
    T_bc = cam.T_body_cam

    con_gt = con_gt and seq.tiene_gt
    if con_gt:
        p_gt, R_gt, _ = seq.gt_en(seq.t_cam[k0:k1])
        T_wc = np.array([make_T(R_gt[i] @ T_bc[:3, :3],
                                R_gt[i] @ T_bc[:3, 3] + p_gt[i])
                         for i in range(k1 - k0)])

    tr = RastreadorKLT(c)
    ids_kf, _, pts_kf = tr.rastrear(seq.leer_imagen(k0), k0)
    k_kf = k0
    medidas, salud = [], []

    for k in range(k0 + 1, k1):
        ids, _, p_cur = tr.rastrear(seq.leer_imagen(k), k)
        salud.append(len(p_cur))
        comunes, i_kf, i_cur = np.intersect1d(ids_kf, ids, return_indices=True)

        forzar = (k - k_kf >= c.kf_max) or (len(comunes) < c.min_comunes)
        if (k - k_kf < c.kf_min) and not forzar:
            continue
        if len(comunes) < 15:
            ids_kf, pts_kf, k_kf = ids.copy(), p_cur.copy(), k     # track roto
            continue

        n_kf = cam.normalizar(pts_kf[i_kf])
        n_cu = cam.normalizar(p_cur[i_cur])
        T_c1c0, inl = pose_relativa(n_kf, n_cu, fm, c.umbral_ransac_px)
        if T_c1c0 is None:
            if forzar:
                ids_kf, pts_kf, k_kf = ids.copy(), p_cur.copy(), k
            continue

        par = float(np.median(paralaje_derotada(n_kf, n_cu, T_c1c0[:3, :3], fm)))
        if par < c.paralaje_min and not forzar:
            continue                       # esperar a que se abra la base

        T_c0c1 = inv_T(T_c1c0)
        u_cam = T_c0c1[:3, 3] / max(np.linalg.norm(T_c0c1[:3, 3]), 1e-12)

        m = dict(k_kf=k_kf, k=k, T_c0c1=make_T(T_c0c1[:3, :3], u_cam),
                 paralaje=par, n_comunes=len(comunes),
                 inliers=float(inl.mean()), bajo_paralaje=par < c.paralaje_min)

        if con_gt:      # verdad de referencia, SOLO diagnóstico
            T_g = inv_T(T_wc[k_kf - k0]) @ T_wc[k - k0]
            s_gt = float(np.linalg.norm(T_g[:3, 3]))
            m["s_gt"] = s_gt
            m["err_rot"] = float(np.degrees(np.linalg.norm(
                Log(T_g[:3, :3].T @ T_c0c1[:3, :3]))))
            cos = (T_g[:3, 3] @ u_cam) / max(s_gt, 1e-12)
            m["err_dir"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        medidas.append(m)
        ids_kf, pts_kf, k_kf = ids.copy(), p_cur.copy(), k

    if verbose and medidas:
        pa = np.array([m["paralaje"] for m in medidas])
        dk = np.array([m["k"] - m["k_kf"] for m in medidas])
        print(f"medidas visuales   : {len(medidas)} en {k1-k0} frames "
              f"(1 cada {dk.mean():.1f} frames)")
        print(f"features por frame : mediana {np.median(salud):.0f}   min {np.min(salud)}")
        print(f"paralaje derotado  : mediana {np.median(pa):.2f} px   "
              f"p10 {np.percentile(pa, 10):.2f}")
        if con_gt:
            er = np.array([m["err_rot"] for m in medidas])
            ed = np.array([m["err_dir"] for m in medidas])
            sg = np.array([m["s_gt"] for m in medidas])
            print(f"error de rotación  : mediana {np.median(er):.3f}°   "
                  f"p90 {np.percentile(er, 90):.3f}°  (contra GT)")
            print(f"error de dirección : mediana {np.median(ed):.2f}°   "
                  f"p90 {np.percentile(ed, 90):.2f}°  (contra GT)")
            print(f"despl. real        : mediana {np.median(sg)*100:.1f} cm   "
                  f"factor max/min {sg.max()/max(sg.min(),1e-9):.1f}x")
    return medidas
