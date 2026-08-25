# Anexo A — Convenios (leer antes de escribir una sola línea)

En VIO el 80 % del tiempo perdido son convenios mal fijados. Este anexo define los
que usa **todo el curso**. Si en algún momento algo "casi funciona pero deriva
raro", vuelve aquí.

---

## A.1 Notación de transformadas

Uso la notación `T_A_B` = transformada que **lleva coordenadas del frame B al frame A**:

```
p_A = T_A_B @ p_B
```

Composición: `T_A_C = T_A_B @ T_B_C` (los subíndices "se cancelan" en el medio).
Inversa: `T_B_A = inv(T_A_B)`.

`T_A_B` también se lee como "**la pose del frame B expresada en A**": su columna de
traslación `t_A_B` es la posición del origen de B vista desde A.

En el código:

```python
T_wc   # pose de la cámara en el mundo  (p_w = T_wc @ p_c)
T_cw   # inversa: matriz de proyección  (p_c = T_cw @ p_w)
R_ws   # rotación body/IMU -> mundo
```

**Nunca** escribas `T_cam_imu` sin decidir si significa `T_cam_from_imu` o
"la pose de la cámara respecto del IMU". Yo uso siempre la primera lectura:
`T_cam_imu ≡ T_C_I`, que mapea puntos del frame IMU al frame cámara.

---

## A.2 Los frames de EuRoC

| Frame | Símbolo EuRoC | Definición |
|---|---|---|
| Mundo / referencia | `R` (reference) o `W` | Fijo, **alineado con la gravedad, z hacia arriba**. Origen definido por el sistema de mocap/Leica. |
| Body | `S` (sensor/body) o `I` | **Coincide con el frame del IMU** (`imu0/sensor.yaml` tiene `T_BS = I₄`). |
| Cámara 0 | `C0` | Óptico estándar: **z hacia adelante, x derecha, y abajo**. |

### El `T_BS` de los YAML

Cada `sensor.yaml` trae una matriz 4×4 `T_BS` (row-major). Su significado es:

```
T_BS = T_body_from_sensor      →      p_body = T_BS @ p_sensor
```

Comprobación de cordura para `cam0` de EuRoC: la traslación de `T_BS` es
aproximadamente `(-0.0216, -0.0640, 0.0098)` m, que es la posición física de la
cámara izquierda respecto del IMU. Si tu lectura de la matriz te da traslaciones de
metros o valores absurdos, la has transpuesto.

Por tanto:

```python
T_body_cam = T_BS_cam0          # tal cual viene del YAML
T_cam_imu  = np.linalg.inv(T_BS_cam0)   # esto es lo que necesitas para proyectar
```

y para proyectar un punto del mundo en la imagen, con `T_ws` la pose del body:

```
p_cam = T_cam_imu @ inv(T_ws) @ p_world
```

### El ground truth

`state_groundtruth_estimate0/data.csv` da 17 columnas:

```
timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z,
           q_RS_w, q_RS_x, q_RS_y, q_RS_z,
           v_RS_R_x, v_RS_R_y, v_RS_R_z,
           b_w_x, b_w_y, b_w_z,
           b_a_x, b_a_y, b_a_z
```

- `p_RS_R` = posición del body S expresada en R → es `t_ws`.
- `q_RS` = rotación de S a R → es `R_ws`. **Escalar primero** (w, x, y, z).
- `v_RS_R` = velocidad del body **expresada en el mundo** (no en body). Importante.
- `b_w`, `b_a` = bias del giróscopo y del acelerómetro, **en frame IMU**.

Que el GT traiga los bias es un regalo: en Fase 3 los usas para inicializar y así
separas el error de integración del error de estimación de bias.

---

## A.3 Cuaterniones: Hamilton vs JPL

Hay dos convenios incompatibles y ambos aparecen en la literatura de VIO:

| | Hamilton | JPL |
|---|---|---|
| Regla `ij` | `ij = k` | `ij = -k` |
| Orden típico en memoria | `(w,x,y,z)` o `(x,y,z,w)` | `(x,y,z,w)` |
| Composición | `q_AC = q_AB ⊗ q_BC` | `q_CA = q_CB ⊗ q_BA` (¡invertida!) |
| Quién lo usa | EuRoC GT, ROS, Eigen, scipy, GTSAM, Solà | Mourikis & Roumeliotis (MSCKF), OpenVINS, JPL/NASA |

**Decisión de este curso: Hamilton en todo, con orden `(x,y,z,w)` internamente
porque es el de `scipy.spatial.transform.Rotation`.** El GT de EuRoC viene en
`(w,x,y,z)`, así que hay que reordenar al leerlo — es un error clásico.

```python
from scipy.spatial.transform import Rotation as Rot
q_wxyz = gt[["qw","qx","qy","qz"]].values
R_ws = Rot.from_quat(q_wxyz[:, [1,2,3,0]])   # scipy quiere (x,y,z,w)
```

Cuando leas el paper del MSCKF o el código de `msckf_tutorial`, **estás en JPL**.
Las consecuencias prácticas:

- La matriz de rotación asociada es la transpuesta de la que esperarías: en JPL,
  `q_IG` representa `R_IG` (mundo→IMU), no `R_GI`.
- El orden de la composición se invierte.
- Los jacobianos cambian de signo en sitios concretos.

No mezcles. Si sigues `msckf_tutorial` (Fase 6A), trabaja en JPL dentro del filtro y
convierte solo en la frontera (entrada de GT, salida a TUM).

---

## A.4 Rotaciones y el mapa exponencial

Para el estado de error usarás siempre un vector de rotación `δθ ∈ R³` con

```
R ← R · Exp(δθ)        (perturbación por la derecha, "local"/body frame)
```

o bien

```
R ← Exp(δθ) · R        (perturbación por la izquierda, "global"/world frame)
```

Ambas son válidas, **pero los jacobianos son distintos**. Solà (arXiv:1711.02508)
desarrolla ambas; el MSCKF clásico usa la local. Este curso usa **local (derecha)**.

```python
def Exp(phi):
    """Vector de rotación (3,) -> matriz SO(3) (3,3)."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + skew(phi)
    axis = phi / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)

def Log(R):
    """SO(3) -> vector de rotación."""
    cos_t = np.clip((np.trace(R)-1)/2, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-10:
        return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/2
    return theta/(2*np.sin(theta)) * np.array([R[2,1]-R[1,2],
                                               R[0,2]-R[2,0],
                                               R[1,0]-R[0,1]])

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
```

---

## A.5 Gravedad y modelo del IMU

El mundo de EuRoC es z-arriba, así que:

```python
g_w = np.array([0.0, 0.0, -9.81])
```

Modelo del acelerómetro (mide **fuerza específica**, no aceleración):

```
a_medida = R_ws^T (a_world - g_w) + b_a + n_a
```

Despejando lo que necesitas para integrar:

```
a_world = R_ws (a_medida - b_a) + g_w
```

**Test de cordura obligatorio**: con el dron quieto sobre la mesa (los primeros
segundos de MH_01), `||a_medida|| ≈ 9.81` y `a_world ≈ 0`. Si te sale `a_world ≈
(0,0,-19.6)` has sumado la gravedad con el signo equivocado. Si te sale
`(0,0,+19.6)`, idem al revés.

Giróscopo:

```
w_medida = w_body + b_g + n_g
```

### Ruidos: de continuo a discreto

`imu0/sensor.yaml` da **densidades espectrales** (continuo). Para el filtro
necesitas covarianzas discretas. La conversión:

```
σ_ruido_blanco_discreto  = σ_density / sqrt(dt)          [rad/s, m/s²]
σ_random_walk_discreto   = σ_rw * sqrt(dt)               [rad/s, m/s²]
```

Valores de EuRoC (`imu0/sensor.yaml`):

```yaml
gyroscope_noise_density:      1.6968e-04   # rad/s/√Hz
gyroscope_random_walk:        1.9393e-05   # rad/s²/√Hz
accelerometer_noise_density:  2.0000e-03   # m/s²/√Hz
accelerometer_random_walk:    3.0000e-03   # m/s³/√Hz
```

Nota práctica: estos valores están **inflados** respecto a lo que da un Allan
variance del ADIS16448. Muchos sistemas (VINS-Mono, OpenVINS) los multiplican por
factores 1–10 para robustez. Empieza con los del YAML y no los toques hasta la
Fase 8.

---

## A.6 Timestamps

- Todos los CSV traen `timestamp [ns]` como entero de 64 bits.
- Conviértelos a segundos en `float64` **restando el primer timestamp** de la
  secuencia. Si no restas el offset, `1.4e18 ns → 1.4e9 s` y con `float64` la
  resolución efectiva cae a ~µs, lo que te destroza los `dt` de 5 ms del IMU.

```python
T0 = min(imu.t_ns.iloc[0], cam.t_ns.iloc[0], gt.t_ns.iloc[0])
imu["t"] = (imu.t_ns - T0).astype(np.float64) * 1e-9
```

**Este es el bug silencioso más común de todo el curso.** Sin el offset todo
"parece" funcionar y el filtro deriva un 10 % más de lo que debería.

- Cámara e IMU están **hardware-sincronizadas** en EuRoC: no hay time offset que
  estimar (a diferencia de tu setup de ATALAYA con una Pi y un autopiloto, donde sí
  lo hay y es de decenas de ms).
- El GT tiene su propia base de tiempo pero ya viene alineada con la de los
  sensores en `state_groundtruth_estimate0`.

---

## A.7 Chuleta de errores frecuentes

| Síntoma | Causa habitual |
|---|---|
| La posición integrada se va a `1e6` en 5 s | Signo de la gravedad, o `a_world` sin rotar |
| La trayectoria estimada es espejo de la GT | Cuaternión leído como `(x,y,z,w)` cuando era `(w,x,y,z)` |
| Deriva de yaw enorme y el resto bien | Bias del giro sin restar |
| Todo bien pero con la escala mal por un factor constante | Estás en monocular sin escala; usa Sim(3) al evaluar |
| El filtro diverge al añadir la cámara | `T_cam_imu` invertida |
| ATE razonable pero RPE horrible | Errores de sincronización / `dt` mal calculado |
| El EKF explota tras N pasos | Covarianza no simetrizada: haz `P = 0.5*(P + P.T)` cada paso |
