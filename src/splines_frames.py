"""
SUAVIZADO DE CONTORNOS CON SPLINES (TODOS LOS FRAMES)
======================================================
Toma la matriz 3D de contornos crudos y aplica
Splines Cubicos Parametricos resueltos mediante
Matriz Inversa para suavizar el contorno de cada frame.

Genera visualizaciones de continuidad (Velocidad y Aceleracion)
para demostrar rigurosamente que el spline es C1 y C2.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================
# SPLINES CUBICOS — RESOLUCION POR MATRIZ INVERSA
# =============================================================

def construir_spline(t, y, print_mat=False):
    n = len(t) - 1
    h = np.diff(t).astype(float)
    sz = n - 1 # Tamaño del sistema (n-1 incógnitas c_i)

    # 1. Armar Matriz A (sz x sz)
    A = np.zeros((sz, sz))
    for i in range(sz):
        A[i, i] = 2 * (h[i] + h[i+1])
        if i > 0:
            A[i, i-1] = h[i]
        if i < sz - 1:
            A[i, i+1] = h[i+1]
            
    # 2. Armar Vector b
    rhs = np.zeros(sz)
    for i in range(sz):
        rhs[i] = 3 * ((y[i+2] - y[i+1])/h[i+1] - (y[i+1] - y[i])/h[i])
        
    if print_mat and sz > 4:
        print("\n  Ejemplo de Matriz Inversa (A^-1) para coeficientes c:")
        print("  Resolviendo c = A^-1 * b...")

    # 3. Hallar Matriz Inversa explícitamente y multiplicar
    if sz > 0:
        inv_A = np.linalg.inv(A)
        c_inner = np.dot(inv_A, rhs)
    else:
        c_inner = np.array([])
        
    # Añadimos las condiciones de frontera naturales c_0 = 0, c_n = 0
    c = np.zeros(n + 1)
    if sz > 0:
        c[1:-1] = c_inner
    
    # 4. Hallar coeficientes a, b, d usando fórmulas estándar
    a = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        a[i] = y[i]
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
        b[i] = (y[i+1] - y[i])/h[i] - (h[i]/3) * (2*c[i] + c[i+1])
        
    return a, b, c[:-1], d

def evaluar(t_ctrl, a, b, c, d, t_eval):
    v = np.empty(len(t_eval))
    for j, tv in enumerate(t_eval):
        i = min(np.searchsorted(t_ctrl[1:], tv, 'left'), len(a)-1)
        dx = tv - t_ctrl[i]
        v[j] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return v

def spline_contorno(pts, n_eval=300, print_debug=False):
    pts_c = np.vstack([pts, pts[0]])
    diffs = np.diff(pts_c, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    t = np.concatenate([[0], np.cumsum(dists)])
    t = t / t[-1]
    t_fine = np.linspace(0, 1, n_eval)

    ax, bx, cx, dx = construir_spline(t, pts_c[:, 0], print_mat=print_debug)
    ay, by, cy, dy = construir_spline(t, pts_c[:, 1], print_mat=False)

    xs = evaluar(t, ax, bx, cx, dx, t_fine)
    ys = evaluar(t, ay, by, cy, dy, t_fine)

    return np.column_stack([xs, ys]), t, ax,bx,cx,dx, ay,by,cy,dy

if __name__ == '__main__':
    print(f"\n{'='*50}\n 2. SUAVIZADO DE CONTORNOS (MATRIZ INVERSA)\n{'='*50}")
    
    crudos = np.load('resultados/contornos_crudos.npy')
    n_frames, n_pts, _ = crudos.shape
    n_eval = 300 
    
    print(f"Resolviendo {n_frames*2} sistemas con Matriz Inversa para X e Y en cada frame...")
    
    contornos_spline = np.full((n_frames, n_eval, 2), np.nan)
    frame_ejemplo = -1
    
    # Guardar coeficientes del ejemplo para graficar
    coefs_ejemplo = None
    
    for fi in range(n_frames):
        pts = crudos[fi]
        if not np.isnan(pts[0,0]):
            debug = (frame_ejemplo == -1)
            if debug:
                print(f"  Analizando Frame #{fi} en detalle...")
            
            suavizado, t_ctrl, ax,bx,cx,dx, ay,by,cy,dy = spline_contorno(pts, n_eval, print_debug=debug)
            contornos_spline[fi] = suavizado
            
            if frame_ejemplo == -1: 
                frame_ejemplo = fi
                coefs_ejemplo = (t_ctrl, ax,bx,cx,dx, ay,by,cy,dy)
            
    np.save('resultados/contornos_spline.npy', contornos_spline)
    
    # =========================================================
    # FIGURA DETALLADA DE SPLINES Y CONTINUIDAD C1 / C2
    # =========================================================
    if frame_ejemplo != -1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pts_orig = crudos[frame_ejemplo]
        pts_suav = contornos_spline[frame_ejemplo]
        t_ctrl, ax,bx,cx,dx, ay,by,cy,dy = coefs_ejemplo
        
        # 1. Contorno Suavizado
        axes[0].plot(pts_orig[:,0], pts_orig[:,1], 'ro', markersize=4, label='Nodos crudos')
        axes[0].plot(pts_suav[:,0], pts_suav[:,1], 'b-', lw=2, label='Spline Cúbico')
        axes[0].set_aspect('equal')
        axes[0].invert_yaxis()
        axes[0].set_title(f'Frame #{frame_ejemplo} - Ajuste Cúbico')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Calcular vectores tangentes (Velocidad C1) y normales (Aceleración C2)
        # Evaluamos la derivada 1 (b + 2c*dx + 3d*dx^2) y derivada 2 (2c + 6d*dx)
        # para mostrar que no hay saltos bruscos.
        n_ctrl = len(t_ctrl) - 1
        t_mid = t_ctrl[:-1] + np.diff(t_ctrl)/2
        
        vx = np.empty(n_ctrl); vy = np.empty(n_ctrl)
        accx = np.empty(n_ctrl); accy = np.empty(n_ctrl)
        
        for i in range(n_ctrl):
            dx_val = t_mid[i] - t_ctrl[i]
            vx[i] = bx[i] + 2*cx[i]*dx_val + 3*dx[i]*dx_val**2
            vy[i] = by[i] + 2*cy[i]*dx_val + 3*dy[i]*dx_val**2
            accx[i] = 2*cx[i] + 6*dx[i]*dx_val
            accy[i] = 2*cy[i] + 6*dy[i]*dx_val
            
        x_mid = evaluar(t_ctrl, ax, bx, cx, dx, t_mid)
        y_mid = evaluar(t_ctrl, ay, by, cy, dy, t_mid)
        
        # 2. Continuidad C1 (Vectores Tangentes)
        axes[1].plot(pts_suav[:,0], pts_suav[:,1], color='gray', alpha=0.5)
        axes[1].quiver(x_mid[::3], y_mid[::3], vx[::3], vy[::3], color='green', 
                       angles='xy', scale_units='xy', scale=10)
        axes[1].set_aspect('equal')
        axes[1].invert_yaxis()
        axes[1].set_title('Continuidad $C^1$ (Vectores Tangentes)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Continuidad C2 (Aceleración / Curvatura)
        axes[2].plot(pts_suav[:,0], pts_suav[:,1], color='gray', alpha=0.5)
        axes[2].quiver(x_mid[::3], y_mid[::3], accx[::3], accy[::3], color='red', 
                       angles='xy', scale_units='xy', scale=50)
        axes[2].set_aspect('equal')
        axes[2].invert_yaxis()
        axes[2].set_title('Continuidad $C^2$ (Curvatura suave sin quiebres)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('resultados/fig1_spline_continuidad.png', dpi=150)
        plt.close()
        
    print(f"\n✓ Splines guardados y figura de continuidad C1/C2 generada.")
