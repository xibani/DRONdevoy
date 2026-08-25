# Fase 6A — Camino del filtro: MSCKF

**Objetivo:** entender el MSCKF a nivel de poder implementarlo, y ejecutarlo sobre
EuRoC.
**Criterio de éxito:** `Edwinem/msckf_tutorial` corriendo sobre MH_01 con ATE
razonable **y** ser capaz de explicar, sin mirar, por qué se proyecta al espacio
nulo de `H_f`.

---

## 6A.1 La idea en un párrafo

Un EKF-SLAM clásico mete los puntos 3D en el estado: con 500 features tienes un
estado de 1500+ dimensiones y una covarianza de 1500² → O(n³) por actualización.
El MSCKF hace lo contrario: **nunca mete las features en el estado**. Mantiene el
estado del IMU más una **ventana de clones de poses de cámara pasadas**. Cuando una
feature deja de verse, la triangula con todas las poses que la vieron, construye el
residuo de reproyección, y **elimina algebraicamente** la posición 3D de la feature
proyectando el residuo al espacio nulo de su jacobiano. Resultado: una restricción
que solo involucra poses, con coste acotado y **consistente**, porque respeta las
correlaciones entre poses (a diferencia de la VO loosely-coupled de la Fase 5).

Es "tightly-coupled" porque el residuo se construye sobre las **medidas de píxel**,
no sobre una pose precocinada.

---

## 6A.2 El estado

**Estado del IMU (nominal, 16 componentes → 15 de error):**

```
x_I = [ q_IG(4), b_g(3), v_G(3), b_a(3), p_G(3) ]
δx_I = [ δθ(3), δb_g(3), δv(3), δb_a(3), δp(3) ]     ∈ R^15
```

Ojo al orden: Mourikis pone la orientación primero. Cada implementación usa el suyo;
lo importante es ser consistente.

**Estado completo con N clones:**

```
X = [ x_I , (q_C1G, p_C1) , (q_C2G, p_C2) , ... , (q_CNG, p_CN) ]
δX ∈ R^(15 + 6N),    P ∈ R^(15+6N × 15+6N)
```

Con N = 10–20 clones, `P` es de 75×75 a 135×135. Coste totalmente manejable.

⚠️ **Convenio JPL**: `q_IG` representa `R_IG` (mundo → IMU). Si vienes de la Fase 5
(Hamilton, `R_ws` = IMU → mundo), es la **transpuesta**. Ver `anexo_convenios.md` §A.3.

---

## 6A.3 Las cuatro operaciones

### 1. Propagación (idéntica a Fase 5)

Integras el IMU sobre los ~10 samples entre frames y propagas
`P_II ← Φ P_II Φᵀ + Q_d`. Las correlaciones con los clones se propagan como
`P_IC ← Φ P_IC` (los clones no cambian: son poses del pasado, congeladas).

```
P = [ P_II   P_IC ]   →   [ Φ P_II Φᵀ + Q   Φ P_IC ]
    [ P_ICᵀ  P_CC ]       [ (Φ P_IC)ᵀ        P_CC   ]
```

Esa estructura es exactamente la del `clone()` de tu ESKF, generalizada a N.

### 2. Aumento del estado (state augmentation)

Al llegar una imagen, clonas la pose de cámara actual, derivada de la del IMU:

```
q_CG = q_CI ⊗ q_IG
p_C  = p_I + R_IGᵀ p_IC          (p_IC: posición de la cámara en frame IMU)
```

y aumentas la covarianza con `J ∈ R^(6 × 15+6N)`:

```
P ← [ I ] P [ I ]ᵀ
    [ J ]   [ J ]
```

donde `J` es el jacobiano de la nueva pose de cámara respecto del estado de error.
Con la extrínseca fija, `J` tiene bloques `R_CI` en la parte de rotación e `I` y
`−R_IGᵀ [p_IC]×` en la de posición.

### 3. Actualización por features (la parte que importa)

Cuando una feature `f_j` deja de trackearse (o el clon más antiguo va a ser
descartado):

**(a) Triangular.** Con las M poses que la vieron, resuelve la posición 3D.
Mourikis usa **parametrización de profundidad inversa** respecto al primer clon:

```
X_f = (1/ρ) · [α, β, 1]ᵀ     expresado en el frame C1
```

y hace Gauss-Newton sobre `(α, β, ρ)`. La razón: `ρ → 0` representa un punto en el
infinito sin singularidad, lo que hace el problema bien condicionado incluso con
poca paralaje. Con XYZ directo, una feature lejana produce un Hessiano casi
singular y una triangulación que se va a kilómetros.

**(b) Residuo.** Para cada observación `i` de la feature `j`:

```
z_ij = [ X_i / Z_i , Y_i / Z_i ] + n      (coords normalizadas)
r_ij = z_ij − ẑ_ij ≈ H_x,ij · δX + H_f,ij · δp_f + n
```

Apilando las M observaciones:

```
r_j = H_x,j δX + H_f,j p̃_f + n_j        con  H_f,j ∈ R^(2M × 3)
```

**(c) Proyección al espacio nulo — el truco.** `p̃_f` es un parámetro que no está en
el estado y que no quieres estimar. Sea `A` una base ortonormal del **espacio nulo
izquierdo** de `H_f,j` (dimensión `2M − 3`). Multiplicando por `Aᵀ`:

```
r_oj = Aᵀ r_j = Aᵀ H_x,j δX + Aᵀ n_j = H_oj δX + n_oj
```

porque `Aᵀ H_f,j = 0`. **La feature ha desaparecido de la ecuación** y te queda una
restricción de `2M−3` dimensiones que solo habla de poses.

Interpretación geométrica: de las `2M` medidas, `3` se "gastan" en determinar dónde
está el punto; las `2M−3` restantes son la información sobre el movimiento. Es la
misma idea que la restricción epipolar generalizada a M vistas.

```python
# En la práctica, con SVD:
U, S, Vt = np.linalg.svd(H_f)          # H_f: (2M, 3)
A = U[:, 3:]                            # (2M, 2M-3): espacio nulo izquierdo
r_o = A.T @ r
H_o = A.T @ H_x
R_o = A.T @ R_meas @ A                  # = sigma^2 * I si el ruido es isotrópico
```

**(d) Gating χ².** Antes de usar la restricción:

```python
S = H_o @ P @ H_o.T + R_o
d2 = r_o @ np.linalg.solve(S, r_o)
if d2 > chi2_table[len(r_o)]:  descartar la feature
```

Esto es el filtro de outliers principal del MSCKF y es **imprescindible**.

**(e) Compresión QR.** Apilando muchas features, `H_o` puede tener miles de filas
para un estado de 135 columnas. Con `H_o = Q R`:

```
r_n = Q₁ᵀ r_o ,  T_H = R₁    (bloque superior triangular, tantas filas como columnas)
```

Reduce el coste de `O(filas³)` a `O(cols³)`. Sin esto el MSCKF no corre en tiempo
real, aunque en Python didáctico puedes saltártelo al principio.

**(f) Actualización EKF estándar** con `T_H`, `r_n`, y ganancia de Kalman; inyección
en el estado nominal (multiplicativa para los cuaterniones) y Joseph para `P`.

### 4. Marginalización

Cuando la ventana llega a N clones, se eliminan algunos (Mourikis descarta 1 de cada
2 de los más antiguos, para mantener paralaje). Marginalizar aquí es trivial:
**borrar las filas y columnas** de `P` correspondientes. Es exacto porque el clon no
aparece en ninguna restricción futura.

---

## 6A.4 Cómo estudiar `Edwinem/msckf_tutorial`

```bash
git clone https://github.com/Edwinem/msckf_tutorial
cd msckf_tutorial
pip install -r requirements.txt
python ./examples/run_on_euroc.py --euroc_folder ~/datasets/euroc/MH_01_easy/mav0 --use_viewer
```

(Comprueba los flags con `--help`: el repo tiene también `--start_timestamp` para
saltarse el tramo estático inicial, que conviene usar.)

**Orden de lectura recomendado** (los nombres de archivo pueden variar; búscalos por
concepto, no por ruta):

| Orden | Qué buscar | Qué preguntarte |
|---|---|---|
| 1 | Definición del **estado** y de la covarianza | ¿Qué orden tienen los bloques? ¿Cuántos clones? |
| 2 | Utilidades de **cuaternión y SO(3)** | ¿JPL o Hamilton? Compruébalo con `q_a ⊗ q_b` |
| 3 | **Propagación** del IMU y de `P` | ¿Φ exacta o serie? ¿Cómo construye `Q_d`? |
| 4 | **Augmentación** del estado | ¿Cuál es exactamente `J`? |
| 5 | **Triangulación** de la feature | ¿Inverse depth? ¿Cuántas iteraciones GN? ¿Qué rechaza? |
| 6 | Construcción de **`H_x`, `H_f`, `r`** | Deriva a mano el jacobiano de `[X/Z, Y/Z]` |
| 7 | **Null-space projection** | ¿SVD o Givens? |
| 8 | **Gating** y **QR** | ¿Qué tabla χ² usa? |
| 9 | **Update** e inyección | ¿Joseph? ¿Reset del error state? |

El repo documenta explícitamente su convenio de transformadas (`frame1_X_frame2`) y
la distinción JPL/Hamilton, y remite al tutorial de Solà. **Léelo antes de tocar el
código.**

**Ejercicio activo (esto es lo que hace que aprendas):** implementa tú
`null_space_projection(H_x, H_f, r)` y `triangulate_inverse_depth(obs, poses)` en un
notebook aparte y compara numéricamente con las del repo sobre los mismos datos. Si
coinciden, has entendido; si no, la diferencia te dice exactamente qué convenio te
falta.

### Alternativas para contrastar

- `uoip/stereo_msckf` — port Python del S-MSCKF (estéreo). Ejecutas
  `python vio.py --view --path .../MH_01_easy`. Al ser estéreo, la escala es
  observable desde el primer frame y converge mucho mejor. Buen contraste.
- `rohiitb/msckf_vio_python` — reimplementación con comparación conceptual
  MSCKF vs ESKF en el README.
- `RBE549 Project 4` (WPI) — starter code con funciones vacías del S-MSCKF y cálculo
  de ATE sobre MH_01. Formato problem-set: si prefieres aprender rellenando huecos
  guiado en vez de leyendo, empieza por aquí.
- `OpenVINS` (C++) — la referencia de producción. Su documentación
  (`docs.openvins.com`) tiene las derivaciones de jacobianos mejor escritas que
  cualquier paper.

---

## 6A.5 Derivación que debes hacer a mano

El jacobiano del modelo de proyección. Con `p_C = [X, Y, Z]ᵀ` el punto en la cámara
y `z = [X/Z, Y/Z]ᵀ`:

```
∂z/∂p_C = (1/Z) [ 1  0  −X/Z ]
                [ 0  1  −Y/Z ]
```

Y `p_C = R_CG (p_f − p_C_G)`, así que:

```
∂p_C/∂δθ_C = [R_CG (p_f − p_C_G)]×   =  [p_C]×      (convenio JPL, perturbación local)
∂p_C/∂δp_C = −R_CG
∂p_C/∂p̃_f  =  R_CG
```

De ahí:

```
H_x,i = ∂z/∂p_C · [ [p_C]× ,  −R_CG ]        (bloque del clon i)
H_f,i = ∂z/∂p_C · R_CG
```

Haz esto en papel una vez. Es media hora y te ahorra días. Verifica numéricamente:

```python
def numeric_jacobian(f, x, eps=1e-6):
    f0 = f(x); J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x); dx[i] = eps
        J[:, i] = (f(x + dx) - f0) / eps
    return J
```

**Comparar jacobianos analíticos con numéricos es la técnica de depuración más
rentable de todo el curso.** Si `‖J_ana − J_num‖ / ‖J_num‖ > 1e-4`, hay un bug.

---

## 6A.6 El problema de la observabilidad (por qué existen FEJ y OC-EKF)

Un VIO tiene **4 direcciones no observables**: las 3 traslaciones globales y la
rotación en **yaw** alrededor de la gravedad (roll y pitch **sí** son observables
gracias a la gravedad). El sistema linealizado del EKF estándar, sin embargo, gana
información espuria en yaw porque los jacobianos se evalúan en estimaciones
distintas en instantes distintos. Consecuencia: el filtro se vuelve
**sobreconfiado** y acaba derivando más de lo que su covarianza dice.

Dos remedios, ambos presentes en OpenVINS:

- **FEJ (First-Estimates Jacobians)**: evalúa siempre los jacobianos en la *primera*
  estimación de cada estado, no en la actual. Barato y efectivo.
- **OC-EKF (Observability-Constrained)**: proyecta los jacobianos para forzar que el
  espacio nulo tenga la dimensión correcta.

En la Fase 6A no lo implementes. Pero cuando tu MSCKF tenga un ATE de 0.4 m y no
baje, la razón probable es esta, y saber que existe te ahorra buscar en el sitio
equivocado.

---

## 6A.7 Trampas específicas del MSCKF

| Trampa | Síntoma |
|---|---|
| Mezclar JPL y Hamilton | El filtro converge en traslación pero la actitud se va |
| Triangular con XYZ en vez de inverse depth | Features a 10⁶ m, actualizaciones absurdas |
| No filtrar por paralaje antes de triangular | Features degeneradas contaminan el update |
| Espacio nulo *derecho* en vez de *izquierdo* | `Aᵀ H_f ≠ 0`; residuos que no cierran |
| `R_o` sin transformar (`Aᵀ R A`) | Covarianza de medida mal escalada |
| No hacer gating χ² | Un outlier tumba el filtro |
| Marginalizar los clones consecutivos más antiguos | Pierdes paralaje; descarta alternos |
| Ventana demasiado corta (N<8) | Poca información multi-vista, deriva alta |

---

## 6A.8 Entregable

1. `msckf_tutorial` ejecutado sobre MH_01 con trayectoria exportada a TUM.
2. ATE/RPE con `evo` (Fase 7).
3. Notebook propio con: tu implementación de `null_space_projection` y de la
   triangulación por inverse depth, verificadas contra las del repo.
4. Los jacobianos `H_x`, `H_f` derivados a mano y **verificados numéricamente**.
5. Un experimento: ATE en función del tamaño de la ventana de clones (N = 5, 10, 20).
