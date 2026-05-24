/**
 * Biomecánica Matemática de las Aves Paceñas
 * Script principal — Plotly charts, sliders, scroll animations
 */

// ═══════════════════════════════════════════════
// PLOTLY THEME (Light Mode)
// ═══════════════════════════════════════════════
const THEME = {
    font:  { color: '#1e293b', family: 'Inter, sans-serif', size: 13 },
    grid:  '#e2e8f0',
    bg:    '#fafbfd',
    paper: '#ffffff',
    red:   '#dc2626',
    blue:  '#2563eb',
    green: '#059669',
    purple:'#7c3aed',
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

// ═══════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();
    initNavHighlight();

    if (typeof globalData === 'undefined') {
        console.error('globalData no definido. Verificar que assets/datos_exportados.js se cargó.');
        return;
    }

    const data = globalData;
    const frames = data.slider_frames;

    // Render initial state
    renderTable(frames[0].raw_x, frames[0].raw_y);
    renderCVChart(frames[0]);
    renderSplineChart(frames[0]);
    renderFourierChart(data.fourier);
    renderPhasePlot(data.phase_plot);

    // Sliders
    initSliders(frames);

    // Lightbox for images
    initLightbox();
});

// ═══════════════════════════════════════════════
// SCROLL ANIMATIONS
// ═══════════════════════════════════════════════
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                obs.unobserve(entry.target);
            }
        });
    }, { threshold: 0.08 });

    document.querySelectorAll('section').forEach(s => observer.observe(s));
}

// ═══════════════════════════════════════════════
// NAVIGATION HIGHLIGHT
// ═══════════════════════════════════════════════
function initNavHighlight() {
    const sections = document.querySelectorAll('section, header');
    const navLinks = document.querySelectorAll('.nav-link');

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === '#' + id);
                });
            }
        });
    }, { rootMargin: '-40% 0px -55% 0px' });

    sections.forEach(s => observer.observe(s));
}

// ═══════════════════════════════════════════════
// SLIDERS
// ═══════════════════════════════════════════════
function initSliders(frames) {
    const slider1 = document.getElementById('slider-1');
    const slider2 = document.getElementById('slider-2');
    const label1 = document.getElementById('slider-label-1');
    const label2 = document.getElementById('slider-label-2');

    const maxIdx = frames.length - 1;
    slider1.max = maxIdx;
    slider2.max = maxIdx;

    function updateAll(idx) {
        const fd = frames[idx];
        const text = `Muestra: ${idx} | Frame: ${fd.frame_real} | t = ${fd.t.toFixed(2)}s`;

        slider1.value = idx;
        slider2.value = idx;
        label1.textContent = text;
        label2.textContent = text;

        document.getElementById('img-mask').src = `assets/frames/mask_${idx}.jpg`;
        renderTable(fd.raw_x, fd.raw_y);
        updateCVChart(fd, idx);
        updateSplineChart(fd);
    }

    slider1.addEventListener('input', e => updateAll(+e.target.value));
    slider2.addEventListener('input', e => updateAll(+e.target.value));
}

// ═══════════════════════════════════════════════
// TABLE
// ═══════════════════════════════════════════════
function renderTable(x, y) {
    const el = document.getElementById('raw-data-table');
    const rows = x.map((xi, i) =>
        `<tr><td>${i}</td><td>${(xi * 2).toFixed(1)}</td><td>${(y[i] * 2).toFixed(1)}</td></tr>`
    ).join('');
    el.innerHTML = `<table><thead><tr><th>#</th><th>X (px)</th><th>Y (px)</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// ═══════════════════════════════════════════════
// CV CHART (points over original photo)
// ═══════════════════════════════════════════════
function renderCVChart(data) {
    Plotly.newPlot('plotly-cv', [{
        x: data.raw_x, y: data.raw_y,
        mode: 'markers', type: 'scatter',
        name: 'Contorno',
        marker: { color: THEME.red, size: 5, symbol: 'circle', line: { color: '#fff', width: 1 } }
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: THEME.font,
        xaxis: { visible: false, range: [0, 960] },
        yaxis: { visible: false, range: [0, 540], autorange: 'reversed' },
        margin: { t: 10, l: 0, r: 0, b: 0 },
        showlegend: false,
        images: [{
            source: 'assets/frames/orig_0.jpg',
            xref: 'x', yref: 'y', x: 0, y: 0,
            sizex: 960, sizey: 540,
            sizing: 'stretch', opacity: 1, layer: 'below'
        }]
    }, PLOTLY_CONFIG);
}

function updateCVChart(data, idx) {
    Plotly.restyle('plotly-cv', { x: [data.raw_x], y: [data.raw_y] });
    Plotly.relayout('plotly-cv', {
        images: [{
            source: `assets/frames/orig_${idx}.jpg`,
            xref: 'x', yref: 'y', x: 0, y: 0,
            sizex: 960, sizey: 540,
            sizing: 'stretch', opacity: 1, layer: 'below'
        }]
    });
}

// ═══════════════════════════════════════════════
// SPLINE CHART (raw points vs spline curve)
// ═══════════════════════════════════════════════
function renderSplineChart(data) {
    Plotly.newPlot('plotly-spline', [
        {
            x: data.raw_x, y: data.raw_y,
            mode: 'markers', type: 'scatter',
            name: 'Puntos Crudos',
            marker: { color: THEME.red, size: 7, opacity: 0.8 }
        },
        {
            x: data.spline_x, y: data.spline_y,
            mode: 'lines', type: 'scatter',
            name: 'Spline Cúbico (A⁻¹)',
            line: { color: THEME.blue, width: 2.5 }
        }
    ], {
        paper_bgcolor: THEME.paper,
        plot_bgcolor: THEME.bg,
        font: THEME.font,
        xaxis: { title: 'X (px)', gridcolor: THEME.grid, zerolinecolor: THEME.grid },
        yaxis: { title: 'Y (px)', gridcolor: THEME.grid, zerolinecolor: THEME.grid, autorange: 'reversed' },
        margin: { t: 20, l: 60, r: 20, b: 50 },
        legend: {
            x: 0.01, y: 0.99,
            bgcolor: 'rgba(255,255,255,0.9)',
            bordercolor: THEME.grid, borderwidth: 1
        }
    }, PLOTLY_CONFIG);
}

function updateSplineChart(data) {
    Plotly.update('plotly-spline', {
        x: [data.raw_x, data.spline_x],
        y: [data.raw_y, data.spline_y]
    });
}

// ═══════════════════════════════════════════════
// FOURIER CHART (H(t) + spectrum)
// ═══════════════════════════════════════════════
function renderFourierChart(data) {
    const maxAmp = Math.max(...data.amps);

    Plotly.newPlot('plotly-fourier', [
        {
            x: data.t, y: data.H_t,
            mode: 'lines+markers', type: 'scatter',
            name: 'Apertura H(t)',
            line: { color: THEME.purple, width: 2 },
            marker: { size: 4, color: THEME.purple },
            xaxis: 'x', yaxis: 'y'
        },
        {
            x: data.freqs, y: data.amps,
            mode: 'lines', type: 'scatter',
            name: 'Espectro FFT',
            line: { color: THEME.green, width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(5,150,105,0.08)',
            xaxis: 'x2', yaxis: 'y2'
        },
        {
            x: [data.f_fundamental],
            y: [maxAmp],
            mode: 'markers+text', type: 'scatter',
            name: 'Frec. Fundamental',
            text: [data.f_fundamental.toFixed(2) + ' Hz'],
            textposition: 'top center',
            textfont: { color: THEME.red, size: 13, family: 'Inter' },
            marker: { color: THEME.red, size: 12, symbol: 'triangle-down' },
            xaxis: 'x2', yaxis: 'y2'
        }
    ], {
        grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
        paper_bgcolor: THEME.paper,
        plot_bgcolor: THEME.bg,
        font: THEME.font,
        margin: { t: 30, l: 65, r: 20, b: 55 },
        showlegend: false,
        xaxis:  { title: 'Tiempo (s)',       gridcolor: THEME.grid, zerolinecolor: THEME.grid },
        yaxis:  { title: 'Altura H(t) [px]', gridcolor: THEME.grid, zerolinecolor: THEME.grid },
        xaxis2: { title: 'Frecuencia (Hz)',  gridcolor: THEME.grid, zerolinecolor: THEME.grid, range: [0, 15] },
        yaxis2: { title: 'Amplitud |Y[k]|',  gridcolor: THEME.grid, zerolinecolor: THEME.grid }
    }, PLOTLY_CONFIG);
}

// ═══════════════════════════════════════════════
// PHASE PLOT (W vs H)
// ═══════════════════════════════════════════════
function renderPhasePlot(data) {
    Plotly.newPlot('plotly-phase', [{
        x: data.W, y: data.H,
        mode: 'lines+markers', type: 'scatter',
        name: 'Ciclo Aleteo',
        line: { color: THEME.green, width: 2 },
        marker: { size: 5, color: THEME.green, opacity: 0.7 }
    }], {
        paper_bgcolor: THEME.paper,
        plot_bgcolor: THEME.bg,
        font: THEME.font,
        xaxis: { title: 'Envergadura W(t) [px]', gridcolor: THEME.grid, zerolinecolor: THEME.grid },
        yaxis: { title: 'Amplitud H(t) [px]',    gridcolor: THEME.grid, zerolinecolor: THEME.grid },
        margin: { t: 20, l: 65, r: 20, b: 55 }
    }, PLOTLY_CONFIG);
}

// ═══════════════════════════════════════════════
// LIGHTBOX (click to zoom images)
// ═══════════════════════════════════════════════
function initLightbox() {
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeBtn = document.getElementById('lightbox-close');

    // Make images in these containers clickable
    const clickableImages = document.querySelectorAll(
        '.bolivia-card-img img, .bio-image-wrapper img, .media-container img'
    );

    clickableImages.forEach(img => {
        img.style.cursor = 'zoom-in';
        img.addEventListener('click', () => {
            lightboxImg.src = img.src;
            lightbox.classList.add('active');
        });
    });

    // Close on X button
    closeBtn.addEventListener('click', () => {
        lightbox.classList.remove('active');
    });

    // Close on backdrop click
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) {
            lightbox.classList.remove('active');
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && lightbox.classList.contains('active')) {
            lightbox.classList.remove('active');
        }
    });
}
