"""ATALAYA — framework VIO del curso, para EuRoC y para tus datos (ArduPilot).

Convenios: mundo z-arriba, G_W=(0,0,-9.81); Hamilton (x,y,z,w) interno;
T_A_B lleva puntos de B a A; perturbación derecha R = R̄·Exp(δθ);
estado de error [δp, δv, δθ, δb_g, δb_a].
"""
__version__ = "0.1.0"

from .geometria import (G_W, Exp, Log, skew, make_T, inv_T, right_jacobian,
                        align_umeyama, ate_rmse, save_tum)
from .sensores import CamaraPinhole, ParamsImu
from .datasets.base import Secuencia
from .frontend import ConfigFrontend, RastreadorKLT, construir_medidas
from .eskf import ESKF, ConfigEskf, cam_a_body, ejecutar_eskf
