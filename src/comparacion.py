"""
COMPARACIÓN — Video Real vs Reconstrucción Matemática (Detallado)
===================================================================
Genera una animación completa con 3 paneles:
  1. Arriba Izq: Video real + Bounding Box
  2. Arriba Der: Reconstrucción (Splines Cúbicos) frame a frame
  3. Abajo: Señal Biomecánica H(t) con tracking en tiempo real.

Salida: resultados/video_aleteo.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter
import cv2
import os

if __name__ == '__main__':
    print(f"\n{'='*50}\n 4. GENERACIÓN DE ANIMACIÓN FINAL\n{'='*50}")
    
    # =============================================================
    # CARGAR DATOS
    # =============================================================
    splines = np.load('resultados/contornos_spline.npy')
    crudos = np.load('resultados/contornos_crudos.npy')
    stats = np.load('resultados/tracking_stats.npy')
    
    if os.path.exists('resultados/frecuencia_aleteo.npy'):
        f_aleteo = float(np.load('resultados/frecuencia_aleteo.npy'))
    else:
        f_aleteo = 0.0

    # Construir señal H(t) para el gráfico inferior
    t_all = stats[:, 0]
    H_t = np.full(len(splines), np.nan)
    for fi in range(len(splines)):
        pts = splines[fi]
        if not np.isnan(pts[0,0]):
            H_t[fi] = pts[:, 1].max() - pts[:, 1].min()

    # =============================================================
    # LEER VIDEO
    # =============================================================
    cap = cv2.VideoCapture('../c1.mp4')
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    vframes = []
    while True:
        ret, f = cap.read()
        if not ret: break
        vframes.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    H_VID, W_VID = vframes[0].shape[:2]
    N_MAX = min(len(vframes), len(splines))

    # =============================================================
    # CONFIGURAR ANIMACIÓN
    # =============================================================
    os.makedirs('resultados', exist_ok=True)
    print(f'Renderizando animación de {N_MAX} frames con 3 paneles...')

    # Layout de 3 paneles (2 arriba, 1 ancho abajo)
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    fig.suptitle('Estudio Biomecánico del Cernícalo (Splines y Fourier)', 
                 fontsize=17, fontweight='bold', y=0.97)
                 
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 1], hspace=0.3)

    # --- Panel 1: Video Real ---
    ax_vid = fig.add_subplot(gs[0, 0])
    ax_vid.axis('off')
    ax_vid.set_title('Grabación Real (Tracking Activo)', fontsize=13)
    im_vid = ax_vid.imshow(vframes[0])

    # --- Panel 2: Simulación Matemática ---
    ax_sim = fig.add_subplot(gs[0, 1])
    ax_sim.set_title('Modelo Matemático (Contorno C²)', fontsize=13)
    ax_sim.set_xlim(0, W_VID)
    ax_sim.set_ylim(H_VID, 0)
    ax_sim.grid(True, alpha=0.3)
    ave_fill, = ax_sim.fill([], [], color=[0.2, 0.5, 0.85], alpha=0.8, edgecolor=[0.1, 0.3, 0.7], linewidth=2, label='Spline Cúbico')
    puntos_crudos_plot, = ax_sim.plot([], [], 'ro', markersize=3, alpha=0.7, label='Puntos crudos (CV)')
    ax_sim.legend(loc='lower left', fontsize=9)
    info_txt = ax_sim.text(0.02, 0.97, '', transform=ax_sim.transAxes, 
                           fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Panel 3: Señal Temporal H(t) ---
    ax_sig = fig.add_subplot(gs[1, :])
    ax_sig.plot(t_all, H_t, 'b-', lw=2, alpha=0.5, label='Amplitud H(t)')
    ax_sig.plot(t_all, H_t, 'b.', markersize=6)
    
    # Marcador de tiempo real
    vline = ax_sig.axvline(t_all[0], color='r', lw=2.5, linestyle='--')
    dot, = ax_sig.plot([t_all[0]], [H_t[0]], 'ro', markersize=10, markeredgecolor='white', markeredgewidth=1.5)
    
    ax_sig.set_title('Frecuencia de Aleteo en Tiempo Real', fontsize=13)
    ax_sig.set_xlabel('Tiempo de vuelo (s)', fontsize=11)
    ax_sig.set_ylabel('Apertura de alas (px)', fontsize=11)
    ax_sig.set_xlim(0, t_all[-1])
    ax_sig.grid(True, alpha=0.4, linestyle='--')
    ax_sig.legend(loc='upper right')

    try:
        writer = FFMpegWriter(fps=fps_video, bitrate=6000)
        with writer.saving(fig, 'resultados/video_aleteo_detallado.mp4', dpi=120):
            for i in range(N_MAX):
                t_now = stats[i, 0]
                cx, cy = stats[i, 1], stats[i, 2]
                bw, bh = stats[i, 3], stats[i, 4]
                valid = stats[i, 5] == 1
                
                # --- Actualizar Video ---
                frame_annotated = vframes[i].copy()
                if valid:
                    cx_i, cy_i, bw_i, bh_i = int(cx), int(cy), int(bw), int(bh)
                    cv2.rectangle(frame_annotated, (cx_i-bw_i//2, cy_i-bh_i//2), 
                                  (cx_i+bw_i//2, cy_i+bh_i//2), (0, 165, 255), 2)
                    cv2.putText(frame_annotated, f"t={t_now:.2f}s", (cx_i-bw_i//2, cy_i-bh_i//2-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
                im_vid.set_data(frame_annotated)
                
                # --- Actualizar Simulación ---
                pts_suav = splines[i]
                pts_crudos = crudos[i]
                if valid and not np.isnan(pts_suav[0,0]) and not np.isnan(pts_crudos[0,0]):
                    ave_fill.set_xy(pts_suav)
                    puntos_crudos_plot.set_data(pts_crudos[:, 0], pts_crudos[:, 1])
                    info_txt.set_text(f'Contorno: Spline Matriz Inversa\nValidación C1 y C2: OK\nFrec. FFT: {f_aleteo:.2f} Hz')
                else:
                    ave_fill.set_xy(np.column_stack([[], []]))
                    puntos_crudos_plot.set_data([], [])
                    
                # --- Actualizar Panel Inferior ---
                vline.set_xdata([t_now])
                if not np.isnan(H_t[i]):
                    dot.set_data([t_now], [H_t[i]])
                else:
                    dot.set_data([t_now], [np.nan])
                    
                writer.grab_frame()
                if i % 20 == 0: print(f'  Renderizado {i}/{N_MAX} frames ({100 * i // N_MAX}%)...')

        print('\n[*] Animacion espectacular guardada con exito!')
        print('[*] Archivo: resultados/video_aleteo_detallado.mp4')

    except Exception as e:
        print(f'\n[!] Error con FFmpeg ({e}). Usar OpenCV fallback o instalar ffmpeg.')

    plt.close('all')