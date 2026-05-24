# Biomecánica Matemática de las Aves Paceñas

Análisis del vuelo del **Cernícalo Americano** (*Falco sparverius*) mediante Splines Cúbicos, Matriz Inversa y Análisis de Fourier.

| | |
|---|---|
| **Estudiante** | Univ. Poma Condori Erick Fernando |
| **Docente** | Lic. Brigida Carvajal Blanco |
| **Materia** | Métodos Numéricos |
| **Universidad** | Universidad Mayor de San Andrés (UMSA) |
| **Gestión** | I / 2026 |

## Cómo ejecutar

### Ver la presentación web

Abrir `index.html` en el navegador (doble clic). No requiere servidor.

### Ejecutar el pipeline Python

```bash
pip install -r requirements.txt

cd src
python tracking_frames.py
python splines_frames.py
python fourier_tiempo.py
python comparacion.py
python exportar_datos_web.py
```

> **Nota:** Se requiere el video `c1.mp4` en la raíz del proyecto y FFmpeg instalado para generar el video comparativo.
