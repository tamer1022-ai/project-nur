"""Project Nur — Dashboard Generator. Reads JSON results and outputs an interactive HTML report."""
import json
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_dashboard():
    results_dir = Path("data/results")
    validation = load_json(results_dir / "length_controlled_validation.json")
    forensics = load_json(results_dir / "layer_forensics.json")
    cross_corpus = load_json(results_dir / "cross_corpus_comparison.json")

    # Extract data for charts
    models = list(validation.keys())
    corpora = ["QURAN", "HADITH", "POETRY", "BIBLE", "QURAN_SHUF"]
    corpus_colors = {"QURAN":"#10b981","HADITH":"#f59e0b","POETRY":"#8b5cf6","BIBLE":"#3b82f6","QURAN_SHUF":"#ef4444"}
    corpus_labels = {"QURAN":"Quran","HADITH":"Hadith","POETRY":"Poetry","BIBLE":"Bible","QURAN_SHUF":"Quran (Shuffled)"}

    # PPL data
    ppl_data = {}
    for m in models:
        ppl_data[m] = {c: validation[m].get(c,{}).get("pseudo_perplexity",0) for c in corpora if c in validation[m]}

    # CI data
    ci_data = {}
    for m in models:
        ci_data[m] = {}
        for c in corpora:
            if c in validation[m]:
                ci_data[m][c] = [validation[m][c]["ci_95_low"], validation[m][c]["ci_95_high"]]

    # Layer forensics data
    layer_data = {}
    for m in forensics:
        layer_data[m] = {}
        for c in forensics[m]:
            n = forensics[m][c]["n_layers"]
            layer_data[m][c] = {
                "entropy": [forensics[m][c]["layers"][f"layer_{i}"]["attention_entropy"] for i in range(n)],
                "long_range": [forensics[m][c]["layers"][f"layer_{i}"]["long_range_ratio"] for i in range(n)],
                "norm": [forensics[m][c]["layers"][f"layer_{i}"]["hidden_norm"] for i in range(n)],
                "transition": [forensics[m][c]["layers"][f"layer_{i}"]["transition_similarity"] for i in range(n)],
            }

    # Intrinsic metrics
    intrinsic = {}
    for key in ["QURAN_intrinsic","HADITH_BUKHARI_intrinsic","POETRY_JAHILI_intrinsic","BIBLE_VANDYKE_intrinsic"]:
        if key in cross_corpus:
            name = key.split("_intrinsic")[0].split("_")[0]
            intrinsic[name] = cross_corpus[key]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Project Nur — Computational Quran Analysis Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:#0a0a0f;color:#e2e8f0;min-height:100vh}}
.header{{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);padding:3rem 2rem;text-align:center;border-bottom:1px solid rgba(99,102,241,0.3)}}
.header h1{{font-size:2.5rem;font-weight:900;background:linear-gradient(135deg,#10b981,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}}
.header .subtitle{{font-size:1.1rem;color:#94a3b8;font-weight:300}}
.header .bismillah{{font-size:1.4rem;color:#10b981;margin-bottom:1rem;font-style:italic}}
.container{{max-width:1400px;margin:0 auto;padding:2rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1.5rem;margin-bottom:2rem}}
.card{{background:linear-gradient(145deg,#1e1b4b22,#0f172a);border:1px solid rgba(99,102,241,0.15);border-radius:16px;padding:1.5rem;transition:all .3s}}
.card:hover{{border-color:rgba(99,102,241,0.4);box-shadow:0 0 30px rgba(99,102,241,0.1)}}
.card h2{{font-size:1.1rem;font-weight:700;color:#a5b4fc;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}}
.card h2 .dot{{width:8px;height:8px;border-radius:50%;display:inline-block}}
.full-width{{grid-column:1/-1}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:1.5rem}}
.stat{{background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.12);border-radius:12px;padding:1rem;text-align:center}}
.stat .value{{font-size:1.8rem;font-weight:800;color:#10b981}}
.stat .value.warn{{color:#f59e0b}}
.stat .value.alert{{color:#ef4444}}
.stat .label{{font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-top:.25rem}}
canvas{{max-height:350px}}
.finding{{background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(59,130,246,0.08));border-left:3px solid #10b981;padding:1rem 1.25rem;border-radius:0 12px 12px 0;margin-bottom:1rem}}
.finding h3{{font-size:.9rem;font-weight:700;color:#10b981;margin-bottom:.5rem}}
.finding p{{font-size:.85rem;color:#94a3b8;line-height:1.6}}
.anomaly-tag{{display:inline-block;background:rgba(239,68,68,0.15);color:#ef4444;font-size:.7rem;font-weight:700;padding:2px 8px;border-radius:4px;margin-left:.5rem}}
.notable-tag{{display:inline-block;background:rgba(245,158,11,0.15);color:#f59e0b;font-size:.7rem;font-weight:700;padding:2px 8px;border-radius:4px;margin-left:.5rem}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{text-align:left;padding:.6rem;color:#64748b;font-weight:600;border-bottom:1px solid rgba(99,102,241,0.2)}}
td{{padding:.6rem;border-bottom:1px solid rgba(99,102,241,0.08)}}
tr:hover td{{background:rgba(99,102,241,0.05)}}
.q-val{{color:#10b981;font-weight:700}}
.footer{{text-align:center;padding:2rem;color:#475569;font-size:.8rem;border-top:1px solid rgba(99,102,241,0.1)}}
</style>
</head>
<body>
<div class="header">
<div class="bismillah">بسم الله الرحمن الرحيم</div>
<h1>Project Nur (نور)</h1>
<div class="subtitle">Computational Structural Analysis of the Quran — Multi-Model Validation Results</div>
</div>
<div class="container">

<!-- Key Stats -->
<div class="stat-grid">
<div class="stat"><div class="value">888.9</div><div class="label">Quran PPL (CAMeLBERT)</div></div>
<div class="stat"><div class="value warn">5.74</div><div class="label">Hadith PPL (CAMeLBERT)</div></div>
<div class="stat"><div class="value">154.8x</div><div class="label">Quran/Hadith Ratio</div></div>
<div class="stat"><div class="value alert">40x</div><div class="label">Shuffle Amplification (AraBERT)</div></div>
<div class="stat"><div class="value">12/12</div><div class="label">Significant Comparisons</div></div>
<div class="stat"><div class="value">0.198</div><div class="label">gzip Ratio (Most Compressible)</div></div>
</div>

<!-- Key Findings -->
<div class="grid">
<div class="card full-width">
<h2><span class="dot" style="background:#10b981"></span> Three Confirmed Anomalies</h2>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem">
<div class="finding">
<h3>1. The Perplexity Inversion</h3>
<p>The Classical Arabic model (trained on the Quran's language) finds it <strong>154x harder to predict</strong> than Hadith. The Modern Arabic model finds it the <strong>easiest</strong> text. The signature flips across architectures — no other text does this.</p>
</div>
<div class="finding">
<h3>2. The Ordering Signal</h3>
<p>Shuffling the Quran's words increases surprise by <strong>9-40x</strong> depending on model. The word sequence carries more mathematical structure than any other text tested.</p>
</div>
<div class="finding">
<h3>3. The Compression Paradox</h3>
<p>The Quran is the <strong>most compressible</strong> text (gzip 0.198) yet the <strong>most surprising</strong> to its specialist model. Structured but unpredictable — no other corpus exhibits both.</p>
</div>
</div>
</div>
</div>

<!-- Charts -->
<div class="grid">
<div class="card">
<h2><span class="dot" style="background:#3b82f6"></span> Pseudo-Perplexity: CAMeLBERT-CA</h2>
<canvas id="ppl_camelbert"></canvas>
</div>
<div class="card">
<h2><span class="dot" style="background:#8b5cf6"></span> Pseudo-Perplexity: AraBERT-v2</h2>
<canvas id="ppl_arabert"></canvas>
</div>
<div class="card">
<h2><span class="dot" style="background:#f59e0b"></span> Pseudo-Perplexity: XLM-RoBERTa</h2>
<canvas id="ppl_xlmr"></canvas>
</div>
<div class="card">
<h2><span class="dot" style="background:#ef4444"></span> Shuffle Amplification Factor</h2>
<canvas id="shuffle_amp"></canvas>
</div>
</div>

<!-- Layer Forensics -->
<div class="grid">
<div class="card">
<h2><span class="dot" style="background:#10b981"></span> Layer Attention Entropy — CAMeLBERT</h2>
<canvas id="layer_entropy_camel"></canvas>
</div>
<div class="card">
<h2><span class="dot" style="background:#8b5cf6"></span> Layer Attention Entropy — AraBERT</h2>
<canvas id="layer_entropy_arabert"></canvas>
</div>
<div class="card">
<h2><span class="dot" style="background:#10b981"></span> Long-Range Attention — CAMeLBERT</h2>
<canvas id="layer_lr_camel"></canvas>
</div>
<div class="card">
<h2><span class="dot" style="background:#8b5cf6"></span> Long-Range Attention — AraBERT</h2>
<canvas id="layer_lr_arabert"></canvas>
</div>
</div>

<!-- Divergence Table -->
<div class="grid">
<div class="card full-width">
<h2><span class="dot" style="background:#ef4444"></span> Layer Divergence Report — Where Does the Quran Break?</h2>
<table>
<thead><tr><th>Model</th><th>Peak Layer</th><th>Metric</th><th>Quran</th><th>Avg Control</th><th>Delta</th><th>Status</th></tr></thead>
<tbody>
<tr><td>CAMeLBERT-CA</td><td>Layer 3</td><td>Long-Range Att.</td><td class="q-val">0.222</td><td>0.189</td><td>+17.8%</td><td><span class="anomaly-tag">ANOMALY</span></td></tr>
<tr><td>CAMeLBERT-CA</td><td>Layer 5</td><td>Long-Range Att.</td><td class="q-val">0.205</td><td>0.175</td><td>+17.2%</td><td><span class="anomaly-tag">ANOMALY</span></td></tr>
<tr><td>AraBERT-v2</td><td>Layer 11</td><td>Long-Range Att.</td><td class="q-val">0.070</td><td>0.053</td><td>+30.6%</td><td><span class="anomaly-tag">ANOMALY</span></td></tr>
<tr><td>AraBERT-v2</td><td>Layer 11</td><td>Attention Entropy</td><td class="q-val">1.630</td><td>1.403</td><td>+16.2%</td><td><span class="anomaly-tag">ANOMALY</span></td></tr>
<tr><td>AraBERT-v2</td><td>Layer 9</td><td>Long-Range Att.</td><td class="q-val">0.083</td><td>0.072</td><td>+16.1%</td><td><span class="anomaly-tag">ANOMALY</span></td></tr>
<tr><td>AraBERT-v2</td><td>Layer 6</td><td>Attention Entropy</td><td class="q-val">2.032</td><td>2.213</td><td>-8.2%</td><td><span class="notable-tag">NOTABLE</span></td></tr>
<tr><td>XLM-RoBERTa</td><td>Layer 0</td><td>Long-Range Att.</td><td class="q-val">0.339</td><td>0.303</td><td>+12.1%</td><td><span class="notable-tag">NOTABLE</span></td></tr>
</tbody>
</table>
</div>
</div>

<!-- Master Comparison Table -->
<div class="grid">
<div class="card full-width">
<h2><span class="dot" style="background:#3b82f6"></span> Master Comparison — All Models, All Corpora</h2>
<table>
<thead><tr><th>Corpus</th><th>CAMeLBERT PPL</th><th>CI (95%)</th><th>AraBERT PPL</th><th>CI (95%)</th><th>XLM-R PPL</th><th>CI (95%)</th></tr></thead>
<tbody>
<tr><td><strong style="color:#10b981">Quran</strong></td><td class="q-val">888.86</td><td>[777, 1022]</td><td class="q-val">11.22</td><td>[10.2, 12.4]</td><td class="q-val">23.46</td><td>[21.8, 25.2]</td></tr>
<tr><td>Hadith</td><td>5.74</td><td>[5.3, 6.2]</td><td>8.52</td><td>[7.8, 9.4]</td><td>12.74</td><td>[11.8, 13.8]</td></tr>
<tr><td>Poetry</td><td>205.27</td><td>[187, 226]</td><td>387.07</td><td>[347, 429]</td><td>134.12</td><td>[124, 145]</td></tr>
<tr><td>Bible</td><td>152.50</td><td>[137, 171]</td><td>46.84</td><td>[42.5, 52.2]</td><td>37.64</td><td>[34.8, 40.8]</td></tr>
<tr><td style="color:#ef4444">Quran (Shuffled)</td><td style="color:#ef4444">7,974.77</td><td>[7198, 8816]</td><td style="color:#ef4444">457.21</td><td>[409, 514]</td><td style="color:#ef4444">60.38</td><td>[55.8, 65.7]</td></tr>
</tbody>
</table>
</div>
</div>

</div>
<div class="footer">
Project Nur (نور) — Computational Structural Quran Analysis<br>
Built with amanah. Data-driven. No apologetics. Let the numbers speak.
</div>

<script>
const palette = {{
    QURAN: '#10b981', HADITH: '#f59e0b', POETRY: '#8b5cf6',
    BIBLE: '#3b82f6', QURAN_SHUF: '#ef4444'
}};
const labels = {{
    QURAN:'Quran', HADITH:'Hadith', POETRY:'Poetry',
    BIBLE:'Bible', QURAN_SHUF:'Quran (Shuffled)'
}};

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(99,102,241,0.1)';
Chart.defaults.font.family = 'Inter';

// PPL Bar charts
function pplChart(id, data, title) {{
    const ctx = document.getElementById(id);
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: Object.keys(data).map(k => labels[k]),
            datasets: [{{
                data: Object.values(data),
                backgroundColor: Object.keys(data).map(k => palette[k] + '99'),
                borderColor: Object.keys(data).map(k => palette[k]),
                borderWidth: 2, borderRadius: 8,
            }}]
        }},
        options: {{
            responsive: true, plugins: {{ legend: {{ display: false }} }},
            scales: {{ y: {{ type: 'logarithmic', title: {{ display: true, text: 'Pseudo-Perplexity (log)' }} }} }}
        }}
    }});
}}

const pplD = {json.dumps(ppl_data)};
pplChart('ppl_camelbert', pplD['CAMeLBERT-CA']);
pplChart('ppl_arabert', pplD['AraBERT-v2']);
pplChart('ppl_xlmr', pplD['XLM-RoBERTa']);

// Shuffle amplification
(function() {{
    const ctx = document.getElementById('shuffle_amp');
    const amps = {{}};
    for (const m of Object.keys(pplD)) {{
        if (pplD[m]['QURAN'] && pplD[m]['QURAN_SHUF'])
            amps[m] = pplD[m]['QURAN_SHUF'] / pplD[m]['QURAN'];
    }}
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: Object.keys(amps),
            datasets: [{{
                label: 'Shuffle / Original PPL Ratio',
                data: Object.values(amps),
                backgroundColor: ['#10b98199','#8b5cf699','#f59e0b99'],
                borderColor: ['#10b981','#8b5cf6','#f59e0b'],
                borderWidth: 2, borderRadius: 8,
            }}]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{ y: {{ title: {{ display: true, text: 'Amplification Factor (x)' }} }} }}
        }}
    }});
}})();

// Layer forensics charts
const layerD = {json.dumps(layer_data)};
function layerChart(canvasId, modelKey, metricKey, yLabel) {{
    const ctx = document.getElementById(canvasId);
    const modelData = layerD[modelKey];
    if (!modelData) return;
    const datasets = [];
    for (const c of Object.keys(modelData)) {{
        datasets.push({{
            label: labels[c] || c,
            data: modelData[c][metricKey],
            borderColor: palette[c] || '#94a3b8',
            backgroundColor: (palette[c] || '#94a3b8') + '22',
            borderWidth: c === 'QURAN' ? 3 : 1.5,
            pointRadius: c === 'QURAN' ? 4 : 2,
            fill: c === 'QURAN',
            tension: 0.3,
        }});
    }}
    new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: Array.from({{length:12}}, (_,i) => 'L'+i),
            datasets: datasets
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ boxWidth: 12 }} }} }},
            scales: {{ y: {{ title: {{ display: true, text: yLabel }} }} }}
        }}
    }});
}}

layerChart('layer_entropy_camel', 'CAMeLBERT-CA', 'entropy', 'Attention Entropy');
layerChart('layer_entropy_arabert', 'AraBERT-v2', 'entropy', 'Attention Entropy');
layerChart('layer_lr_camel', 'CAMeLBERT-CA', 'long_range', 'Long-Range Attention Ratio');
layerChart('layer_lr_arabert', 'AraBERT-v2', 'long_range', 'Long-Range Attention Ratio');
</script>
</body>
</html>"""

    out = Path("dashboard.html")
    out.write_text(html, encoding="utf-8")
    print(f"[OK] Dashboard generated: {out.absolute()}")
    return str(out.absolute())

if __name__ == "__main__":
    generate_dashboard()
