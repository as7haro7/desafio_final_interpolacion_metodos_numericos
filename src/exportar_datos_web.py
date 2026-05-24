import numpy as np
import json
import os
import cv2

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
VIDEO = '../c1.mp4'
N_MUESTRAS = 20 # Cuántos frames extraeremos para el slider
UMBRAL_AVE = 114

crudos = np.load('resultados/contornos_crudos.npy')
splines = np.load('resultados/contornos_spline.npy')
stats = np.load('resultados/tracking_stats.npy')

n_frames_total = len(splines)
valid_indices = [i for i in range(n_frames_total) if stats[i, 5] == 1 and not np.isnan(splines[i,0,0])]

# Seleccionar N índices equiespaciados
paso = max(1, len(valid_indices) // N_MUESTRAS)
selected_indices = valid_indices[::paso][:N_MUESTRAS]

# Crear directorios
os.makedirs('../assets/frames', exist_ok=True)

# =====================================================================
# EXTRAER IMÁGENES Y DATOS DEL SLIDER
# =====================================================================
cap = cv2.VideoCapture(VIDEO)

frames_data = []

for idx, fi in enumerate(selected_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret: continue
    
    # Obtener máscara
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, UMBRAL_AVE)
    k3 = np.ones((3,3), np.uint8)
    k7 = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=3)
    
    # Recortar alrededor del ave para mejor visualización
    cx, cy = stats[fi, 1], stats[fi, 2]
    w, h = stats[fi, 3], stats[fi, 4]
    
    # Guardar imagen original completa (redimensionada para no pesar mucho)
    frame_small = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    cv2.imwrite(f'../assets/frames/orig_{idx}.jpg', frame_small)
    
    mask_small = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2))
    cv2.imwrite(f'../assets/frames/mask_{idx}.jpg', mask_small)
    
    # Añadir a la lista de JSON
    frames_data.append({
        "id": idx,
        "frame_real": fi,
        "t": stats[fi, 0],
        "cx": cx, "cy": cy, "w": w, "h": h,
        "raw_x": [p[0]/2 for p in crudos[fi]], # Dividir por 2 porque redimensionamos la imagen a la mitad
        "raw_y": [p[1]/2 for p in crudos[fi]],
        "spline_x": [p[0]/2 for p in splines[fi]],
        "spline_y": [p[1]/2 for p in splines[fi]]
    })

cap.release()

# =====================================================================
# EXTRAER SEÑAL FOURIER Y DIAGRAMA DE FASE
# =====================================================================
t_raw = stats[:, 0]
fps = 1.0 / np.nanmean(np.diff(t_raw)) if n_frames_total > 1 else 30.0

H_t = np.full(n_frames_total, np.nan)
W_t = np.full(n_frames_total, np.nan)

for fi in range(n_frames_total):
    pts = splines[fi]
    if not np.isnan(pts[0,0]):
        H_t[fi] = pts[:, 1].max() - pts[:, 1].min()
        W_t[fi] = pts[:, 0].max() - pts[:, 0].min()
        
t_clean = np.linspace(t_raw[0], t_raw[-1], n_frames_total)
mask_v = ~np.isnan(H_t)

H_interp = np.interp(t_clean, t_clean[mask_v], H_t[mask_v])

sig = H_interp - np.mean(H_interp)
N_f = len(sig)
Y = np.fft.rfft(sig)
freqs = np.fft.rfftfreq(N_f, d=1/fps)
amps = np.abs(Y) / N_f

mask_f = (freqs >= 1) & (freqs <= 20)
f_aleteo = freqs[mask_f][np.argmax(amps[mask_f])] if np.sum(mask_f) > 0 else 0

# Construir JSON final
data_export = {
    "slider_frames": frames_data,
    "fourier": {
        "t": t_clean.tolist(),
        "H_t": H_interp.tolist(),
        "freqs": freqs[mask_f].tolist(),
        "amps": amps[mask_f].tolist(),
        "f_fundamental": float(f_aleteo)
    },
    "phase_plot": {
        "H": H_t[mask_v].tolist(),
        "W": W_t[mask_v].tolist()
    }
}

with open('../assets/datos_exportados.js', 'w') as f:
    f.write('const globalData = ' + json.dumps(data_export) + ';')

print(f"Datos exportados exitosamente. {len(frames_data)} frames guardados en JS.")

# =====================================================================
# COPIAR FIGURAS Y VIDEO A ASSETS (para la web)
# =====================================================================
import shutil

archivos_web = [
    ('resultados/fig1_spline_continuidad.png', '../assets/fig1_spline_continuidad.png'),
    ('resultados/video_aleteo_detallado.mp4', '../assets/video_aleteo_detallado.mp4'),
]

for src, dst in archivos_web:
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copiado: {src} -> {dst}")
    else:
        print(f"  [!] No encontrado: {src} (ejecutar el script correspondiente primero)")
