"""
ANÁLISIS DE FOURIER EN EL TIEMPO
=================================
Toma los contornos suavizados de todos los frames
y extrae la "Altura" del ave en función del tiempo H(t).
Aplica la Transformada Rápida de Fourier (FFT) a esta
señal 1D para hallar la frecuencia temporal de aleteo.

Genera una figura rica en explicaciones y
muestra una tabla detallada de frecuencias dominantes.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    print(f"\n{'='*50}\n 3. ANÁLISIS DE FRECUENCIA DE ALETEO (FOURIER FFT)\n{'='*50}")
    
    splines = np.load('resultados/contornos_spline.npy')
    stats = np.load('resultados/tracking_stats.npy')
    
    t_raw = stats[:, 0]
    valid = stats[:, 5] == 1
    
    n_frames = len(splines)
    fps = 1.0 / np.nanmean(np.diff(t_raw)) if n_frames > 1 else 30.0
    
    # 1. Señal H(t)
    H_t = np.full(n_frames, np.nan)
    for fi in range(n_frames):
        pts = splines[fi]
        if not np.isnan(pts[0,0]):
            H_t[fi] = pts[:, 1].max() - pts[:, 1].min()
            
    t_clean = np.linspace(t_raw[0], t_raw[-1], n_frames)
    mask = ~np.isnan(H_t)
    
    if np.sum(mask) == 0:
        raise ValueError("No hay datos válidos para procesar.")
        
    H_interp = np.interp(t_clean, t_clean[mask], H_t[mask])
    
    # 2. FFT
    sig = H_interp - np.mean(H_interp)
    N = len(sig)
    
    Y = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(N, d=1/fps)
    amps = np.abs(Y) / N
    
    mask_f = (freqs >= 1) & (freqs <= 20)
    f_aleteo = 0.0
    
    print(f"\nAnalizando señal temporal de {N} frames muestreada a {fps:.2f} Hz...")
    print(f"Top 5 Frecuencias Dominantes (Espectro > 1Hz):")
    print("-" * 50)
    print(f"| Rango | Frecuencia (Hz) | Amplitud |")
    print("-" * 50)
    
    if np.sum(mask_f) > 0:
        freqs_valid = freqs[mask_f]
        amps_valid = amps[mask_f]
        idx_top = np.argsort(amps_valid)[::-1][:5]
        
        for i, idx in enumerate(idx_top):
            f_val = freqs_valid[idx]
            a_val = amps_valid[idx]
            if i == 0: f_aleteo = f_val
            print(f"|  #{i+1}  |    {f_val:5.2f} Hz     |  {a_val:7.2f} |")
    print("-" * 50)
        
    np.save('resultados/frecuencia_aleteo.npy', np.array(f_aleteo))
    
    # =========================================================
    # FIGURA DETALLADA FOURIER
    # =========================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    
    # Señal Temporal H(t)
    axes[0].plot(t_clean, H_interp, 'b-', lw=2.5, alpha=0.8, label='Altura Interpolada $H(t)$')
    axes[0].plot(t_clean[mask], H_t[mask], 'r.', markersize=8, label='Datos Reales (Splines)')
    # Superponer una senoide ajustada a la frecuencia fundamental
    A = amps_valid[idx_top[0]] * 2 if np.sum(mask_f)>0 else 0
    y_sin = np.mean(H_interp) + A * np.cos(2*np.pi*f_aleteo*t_clean - np.angle(Y[mask_f][idx_top[0]]))
    axes[0].plot(t_clean, y_sin, 'g--', lw=2, label=f'Onda Fundamental ({f_aleteo:.2f} Hz)')
    
    axes[0].set_title('Dinámica del Vuelo: Amplitud Vertical del Ave $H(t)$', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tiempo de vuelo (s)', fontsize=12)
    axes[0].set_ylabel('Altura del Contorno (Píxeles)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.4, linestyle='--')
    
    # Espectro Fourier
    axes[1].plot(freqs, amps, 'b-', lw=2)
    axes[1].fill_between(freqs, 0, amps, color='blue', alpha=0.1)
    
    # Marcar top frecuencias
    if np.sum(mask_f) > 0:
        for i, idx in enumerate(idx_top):
            f_val = freqs_valid[idx]
            a_val = amps_valid[idx]
            color = 'red' if i == 0 else 'purple'
            axes[1].plot(f_val, a_val, marker='v', color=color, markersize=8)
            axes[1].annotate(f'{f_val:.2f} Hz', (f_val, a_val), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontsize=11, color=color, fontweight='bold')
                             
    axes[1].axvspan(5.0, 7.5, color='orange', alpha=0.15, label='Rango biológico típico cernícalos (5-7.5 Hz)')
    axes[1].set_xlim(0, 15)
    axes[1].set_ylim(0, max(amps)*1.2)
    axes[1].set_title('Espectro de Frecuencias (Transformada de Fourier)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frecuencia del Aleteo (Hz)', fontsize=12)
    axes[1].set_ylabel('Magnitud / Dominancia', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('resultados/fig2_fourier_detallado.png', dpi=150)
    plt.close()
    
    print("\n✓ Análisis completado.")
    print("✓ Gráfico detallado generado: resultados/fig2_fourier_detallado.png")
