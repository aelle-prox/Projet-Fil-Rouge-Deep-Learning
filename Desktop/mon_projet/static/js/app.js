/* ═══════════════════════════════════════════════════════════════
   NAKLES — JavaScript principal
   ═══════════════════════════════════════════════════════════════ */

/* ── Toast ─────────────────────────────────────────────────── */
function showToast(message, type = 'success') {
  const old = document.querySelector('.toast');
  if (old) old.remove();

  const icon = type === 'success' ? '✅' : '❌';
  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  document.body.appendChild(toast);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add('show'));
  });

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, 3500);
}

/* ── Lien actif navbar ─────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  document.querySelectorAll('.navbar__links a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
});

/* ── Compteur animé ────────────────────────────────────────── */
function animateCounter(el, target, decimals = 0, suffix = '') {
  const duration = 1200;
  const start = performance.now();
  const startVal = 0;

  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const val = startVal + (target - startVal) * ease;
    el.textContent = val.toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── Intersection Observer pour animations ─────────────────── */
const observers = {};
function observeOnce(selector, callback) {
  const els = document.querySelectorAll(selector);
  if (!els.length) return;
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        callback(entry.target);
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2 });
  els.forEach(el => obs.observe(el));
}

/* ── CNN Page ──────────────────────────────────────────────── */
function initCNNPage() {
  const uploadZone   = document.getElementById('uploadZone');
  const fileInput    = document.getElementById('fileInput');
  const previewWrap  = document.getElementById('previewWrap');
  const previewImg   = document.getElementById('previewImg');
  const predictBtn   = document.getElementById('predictBtn');
  const placeholder  = document.getElementById('resultPlaceholder');
  const resultContent = document.getElementById('resultContent');
  const loadingState = document.getElementById('loadingState');

  if (!uploadZone) return;

  let selectedFile = null;

  // Drag & drop
  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      showToast('Veuillez sélectionner une image (PNG, JPG)', 'error');
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewWrap.style.display = 'block';
      predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Afficher loading
    placeholder.style.display   = 'none';
    resultContent.style.display = 'none';
    loadingState.style.display  = 'flex';
    predictBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const resp = await fetch('/api/predict/cnn', {
        method: 'POST',
        body: formData
      });

      if (!resp.ok) throw new Error(`Erreur ${resp.status}`);
      const data = await resp.json();

      // Afficher le résultat
      loadingState.style.display  = 'none';
      resultContent.style.display = 'flex';

      document.getElementById('resultEmoji').textContent = data.emoji;
      document.getElementById('resultNom').textContent   = data.classe_predite;
      document.getElementById('resultConf').textContent  =
        `Confiance : ${(data.confiance * 100).toFixed(1)}%`;

      // Barres de probabilités
      const list = document.getElementById('probaList');
      list.innerHTML = '';

      const sorted = Object.entries(data.toutes_probabilites)
        .sort((a, b) => b[1] - a[1]);
      const topClass = sorted[0][0];

      sorted.forEach(([name, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const isTop = name === topClass;
        list.innerHTML += `
          <div class="proba-item">
            <span class="proba-item__name">${name}</span>
            <div class="proba-bar-wrap">
              <div class="proba-bar ${isTop ? 'top' : ''}"
                   style="width:${pct}%"></div>
            </div>
            <span class="proba-item__val">${pct}%</span>
          </div>`;
      });

      const modeTag = document.getElementById('modeTag');
      if (modeTag) {
        modeTag.textContent = data.mode === 'demo' ? '⚡ Mode démo' : '🟢 Modèle réel';
      }

      showToast(`Classe détectée : ${data.classe_predite}`, 'success');

    } catch (err) {
      loadingState.style.display = 'none';
      placeholder.style.display  = 'flex';
      showToast('Erreur lors de la prédiction', 'error');
      console.error(err);
    } finally {
      predictBtn.disabled = false;
    }
  });
}

/* ── LSTM Page ─────────────────────────────────────────────── */
function initLSTMPage() {
  const inputs     = document.querySelectorAll('.temp-input');
  const predictBtn = document.getElementById('lstmPredictBtn');
  const randomBtn  = document.getElementById('lstmRandomBtn');
  const resultBox  = document.getElementById('lstmResult');
  const placeholder = document.getElementById('lstmPlaceholder');

  if (!inputs.length) return;

  // Remplir avec valeurs aléatoires réalistes
  function fillRandom() {
    let base = Math.random() * 15 - 5; // -5 à 10°C
    inputs.forEach(inp => {
      base += (Math.random() - 0.5) * 1.5;
      base = Math.max(-15, Math.min(35, base));
      inp.value = base.toFixed(1);
    });
    showToast('Séquence générée aléatoirement', 'success');
  }

  if (randomBtn) randomBtn.addEventListener('click', fillRandom);

  // Remplir avec des valeurs initiales
  fillRandom();

  predictBtn.addEventListener('click', async () => {
    const temps = Array.from(inputs).map(inp => {
      const v = parseFloat(inp.value);
      return isNaN(v) ? 0 : v;
    });

    if (temps.length !== 24) {
      showToast('Exactement 24 valeurs requises', 'error');
      return;
    }

    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner" style="width:16px;height:16px;margin:0"></span> Calcul...';

    try {
      const resp = await fetch('/api/predict/lstm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temperatures: temps })
      });

      if (!resp.ok) throw new Error(`Erreur ${resp.status}`);
      const data = await resp.json();

      // Afficher résultat
      if (placeholder) placeholder.style.display = 'none';
      if (resultBox)   resultBox.style.display   = 'flex';

      const predEl = document.getElementById('lstmPredValue');
      if (predEl) {
        predEl.textContent = '...';
        setTimeout(() => {
          predEl.textContent = data.prediction.toFixed(1);
        }, 100);
      }

      // Mini description
      const last = temps[temps.length - 1];
      const diff = data.prediction - last;
      const trend = diff > 0.3 ? '📈 Hausse' : diff < -0.3 ? '📉 Baisse' : '➡️ Stable';
      const trendEl = document.getElementById('lstmTrend');
      if (trendEl) trendEl.textContent = `${trend} (${diff > 0 ? '+' : ''}${diff.toFixed(1)}°C)`;

      // Dessiner mini graphique
      drawMiniChart(temps, data.prediction);

      const modeTag = document.getElementById('lstmModeTag');
      if (modeTag) modeTag.textContent = data.mode === 'demo' ? '⚡ Mode démo' : '🟢 Modèle réel';

      showToast(`Prédiction : ${data.prediction.toFixed(1)}°C`, 'success');

    } catch (err) {
      showToast('Erreur lors de la prédiction LSTM', 'error');
      console.error(err);
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = '⚡ Prédire T+1';
    }
  });
}

/* ── Mini graphique canvas ─────────────────────────────────── */
function drawMiniChart(sequence, prediction) {
  const canvas = document.getElementById('seqCanvas');
  if (!canvas) return;

  const ctx    = canvas.getContext('2d');
  const W      = canvas.width;
  const H      = canvas.height;
  const allVals = [...sequence, prediction];
  const minV   = Math.min(...allVals) - 1;
  const maxV   = Math.max(...allVals) + 1;
  const range  = maxV - minV || 1;

  ctx.clearRect(0, 0, W, H);

  const toX = i  => (i / (allVals.length - 1)) * (W - 40) + 20;
  const toY = v  => H - ((v - minV) / range) * (H - 30) - 15;

  // Grille légère
  ctx.strokeStyle = 'rgba(37,99,235,0.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = (i / 4) * (H - 30) + 15;
    ctx.beginPath();
    ctx.moveTo(20, y);
    ctx.lineTo(W - 20, y);
    ctx.stroke();
  }

  // Ligne séquence
  ctx.beginPath();
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  sequence.forEach((v, i) => {
    i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v));
  });
  ctx.stroke();

  // Ligne prédiction (tiretée)
  const lastIdx = sequence.length - 1;
  ctx.beginPath();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = '#93c5fd';
  ctx.lineWidth = 2;
  ctx.moveTo(toX(lastIdx), toY(sequence[lastIdx]));
  ctx.lineTo(toX(lastIdx + 1), toY(prediction));
  ctx.stroke();
  ctx.setLineDash([]);

  // Point prédiction
  ctx.beginPath();
  ctx.arc(toX(lastIdx + 1), toY(prediction), 5, 0, Math.PI * 2);
  ctx.fillStyle = '#2563eb';
  ctx.fill();
}

/* ── Page métriques ────────────────────────────────────────── */
async function initMetriquesPage() {
  try {
    const resp = await fetch('/api/metriques');
    const data = await resp.json();
    renderMetriques(data);
  } catch (e) {
    console.warn('Métriques API non disponible, affichage statique');
  }
}

function renderMetriques(data) {
  const accEl = document.getElementById('m-cnn-acc');
  const maeEl = document.getElementById('m-lstm-mae');
  if (accEl) animateCounter(accEl, data.cnn.accuracy * 100, 1, '%');
  if (maeEl) animateCounter(maeEl, data.lstm.mae, 4, '');
}

/* ── Init selon la page ────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;

  if (path === '/' || path === '/index') {
    // Compteurs animés sur la home
    observeOnce('.model-card-mini', el => {
      const vals = el.querySelectorAll('.stat-item__val[data-count]');
      vals.forEach(v => {
        const target = parseFloat(v.dataset.count);
        const dec    = v.dataset.dec || 0;
        const suf    = v.dataset.suf || '';
        animateCounter(v, target, parseInt(dec), suf);
      });
    });
  }

  if (path === '/cnn')        initCNNPage();
  if (path === '/lstm')       initLSTMPage();
  if (path === '/metriques')  initMetriquesPage();
});
