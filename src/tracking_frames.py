"""
TRACKING DE CONTORNOS EN TODOS LOS FRAMES
==========================================
Procesa el video `c1.mp4` frame a frame.
Extrae el contorno del ave, su caja delimitadora (Bounding Box),
y estandariza cada contorno a exactamente N puntos (100).

Genera visualizaciones detalladas del tracking:
  - resultados/fig0_tracking.png : Trayectoria y dimensiones.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

VIDEO = '../c1.mp4'
UMBRAL_AVE = 114
N_PUNTOS = 100

def segmentar_ave(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, UMBRAL_AVE)
    k3 = np.ones((3,3), np.uint8)
    k7 = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k3, iterations=1)
    return mask

def remuestrear_contorno(contorno, n_puntos):
    pts = contorno.reshape(-1, 2).astype(float)
    pts = np.vstack([pts, pts[0]])
    diffs = np.diff(pts, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    t = np.concatenate([[0], np.cumsum(dists)])
    if t[-1] == 0: return np.full((n_puntos, 2), np.nan)
    t = t / t[-1]
    
    t_new = np.linspace(0, 1, n_puntos + 1)[:-1]
    x_new = np.interp(t_new, t, pts[:, 0])
    y_new = np.interp(t_new, t, pts[:, 1])
    return np.column_stack([x_new, y_new])

if __name__ == '__main__':
    print(f"\n{'='*50}\n 1. TRACKING FRAME A FRAME\n{'='*50}")
    
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened(): raise FileNotFoundError(f"Video {VIDEO} no encontrado.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W_VID = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_VID = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {W_VID}x{H_VID} @ {fps} FPS | Total: {total_frames} frames\n")

    contornos_all = np.full((total_frames, N_PUNTOS, 2), np.nan)
    stats_all = np.full((total_frames, 6), np.nan) # [t, cx, cy, w, h, valid]
    
    for fi in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        mask = segmentar_ave(frame)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        t_now = fi / fps
        
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                pts_resampled = remuestrear_contorno(cnt, N_PUNTOS)
                contornos_all[fi] = pts_resampled
                
                M = cv2.moments(cnt)
                cx = M['m10']/M['m00'] if M['m00']>0 else np.nan
                cy = M['m01']/M['m00'] if M['m00']>0 else np.nan
                x, y, w, h = cv2.boundingRect(cnt)
                
                stats_all[fi] = [t_now, cx, cy, w, h, 1]
                if fi % 20 == 0:
                    print(f"  Frame {fi:>3} | t={t_now:.2f}s | Ave detectada en ({int(cx):>4}, {int(cy):>4}) | W/H = {w/h:.2f}")
                continue
                
        stats_all[fi] = [t_now, np.nan, np.nan, np.nan, np.nan, 0]

    cap.release()
    
    os.makedirs('resultados', exist_ok=True)
    np.save('resultados/contornos_crudos.npy', contornos_all)
    np.save('resultados/tracking_stats.npy', stats_all)
    
    # =========================================================
    # FIGURA DETALLADA DEL TRACKING
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    t = stats_all[:, 0]
    cx = stats_all[:, 1]
    cy = stats_all[:, 2]
    w = stats_all[:, 3]
    h = stats_all[:, 4]
    
    # Panel 1: Trayectoria
    sc = axes[0].scatter(cx, cy, c=t, cmap='viridis', s=20)
    axes[0].plot(cx, cy, 'k-', alpha=0.3)
    axes[0].set_xlim(0, W_VID)
    axes[0].set_ylim(H_VID, 0)
    axes[0].set_title('Trayectoria del Centroide del Ave', fontsize=13)
    axes[0].set_xlabel('X (px)')
    axes[0].set_ylabel('Y (px)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label='Tiempo (s)')
    
    # Panel 2: Dimensiones
    axes[1].plot(t, w, 'b-', lw=2, label='Ancho (W) - Alas')
    axes[1].plot(t, h, 'r-', lw=2, label='Alto (H) - Aleteo')
    axes[1].set_title('Dimensiones del Bounding Box', fontsize=13)
    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('Píxeles')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados/fig0_tracking_detallado.png', dpi=150)
    plt.close()
    
    print(f"\n✓ Tracking completo guardado.")
    print(f"✓ Gráfico generado: resultados/fig0_tracking_detallado.png")
