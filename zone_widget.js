/* Zone map: where final rings end on each map, and where they're likely to end once ring 1 / ring 2 are known.
   Data: placement.gg ring archive (current ring settings) + landing-spot outlines from apexlegendsstatus.com.
   ZoneWidget(rootElement, ZD, {teams: {team: {map: spot}}, compact: bool}) */
function ZoneWidget(root, ZD, opt) {
  opt = opt || {};
  const B = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_', BI = {};
  [...B].forEach((c, i) => BI[c] = i);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const pct = x => x > 0.99 && x < 1 ? '99%' : (100 * x).toFixed(x < 0.1 ? 1 : 0) + '%';
  const MAPS = Object.keys(ZD.maps), cache = {}, uid = Math.random().toString(36).slice(2, 8);
  const S = {map: MAPS[0], r1: null, r2: null, algs: false, teams: true};

  function inside(x, y, poly) { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [x1, y1] = poly[i], [x2, y2] = poly[j]; if ((y1 > y) !== (y2 > y) && x < x1 + (y - y1) * (x2 - x1) / (y2 - y1)) c = !c; } return c; }
  function spotOf(x, y, outl) {
    for (const [n, p] of outl) if (inside(x, y, p)) return n;
    let best = null, bd = 0.05;
    for (const [n, p] of outl) for (const [a, b] of p) { const d = Math.hypot(a - x, b - y); if (d < bd) { bd = d; best = n; } }
    return best;
  }
  function data(name) {
    if (cache[name]) return cache[name];
    const m = ZD.maps[name], n = m.n, s = m.seq, r1 = new Float32Array(2 * n), r2 = new Float32Array(2 * n), e = new Float32Array(2 * n);
    for (let k = 0; k < n; k++) {
      const c = i => BI[s[7 * k + i]];
      r1[2 * k] = c(0) / 63; r1[2 * k + 1] = c(1) / 63; r2[2 * k] = c(2) / 63; r2[2 * k + 1] = c(3) / 63;
      const v = c(4) * 4096 + c(5) * 64 + c(6); e[2 * k] = Math.floor(v / 512) / 511; e[2 * k + 1] = (v % 512) / 511;
    }
    const algs = new Uint8Array(n); (m.algs ? m.algs.split(',') : []).forEach(i => algs[+i] = 1);
    const spot = []; for (let k = 0; k < n; k++) spot.push(spotOf(e[2 * k], e[2 * k + 1], m.outlines));
    const cent = Object.fromEntries(m.outlines.map(([nm, p]) => [nm, [p.reduce((a, q) => a + q[0], 0) / p.length, p.reduce((a, q) => a + q[1], 0) / p.length]]));
    return cache[name] = {m, n, r1, r2, e, algs, spot, cent};
  }
  function weights(d) {
    const w = new Float64Array(d.n);
    for (let k = 0; k < d.n; k++) {
      if (S.algs && !d.algs[k]) continue;
      let q = 0;
      if (S.r2) { const a = d.r2[2 * k] - S.r2[0], b = d.r2[2 * k + 1] - S.r2[1], c = d.r1[2 * k] - S.r1[0], f = d.r1[2 * k + 1] - S.r1[1];
        q = (a * a + b * b + 0.25 * (c * c + f * f)) / (2 * 0.03 * 0.03); }
      else if (S.r1) { const c = d.r1[2 * k] - S.r1[0], f = d.r1[2 * k + 1] - S.r1[1]; q = (c * c + f * f) / (2 * 0.04 * 0.04); }
      w[k] = Math.exp(-q);
    }
    return w;
  }

  if (!document.getElementById('zw-css')) document.head.insertAdjacentHTML('beforeend', `<style id="zw-css">
    .zw{--zw-a:var(--teal,var(--bar,#2f6fb3));--zw-hot:var(--claret,var(--hot,#c9271f))}
    .zw-grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:22px;align-items:start}
    @media(max-width:820px){.zw-grid{grid-template-columns:1fr}}
    .zw-map{position:relative;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:6px}
    .zw-map svg{display:block;width:100%;height:auto;cursor:crosshair;touch-action:manipulation}
    .zw-ctl{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin:0 0 10px}.zw-ctl>div{flex:1 1 170px;max-width:240px}
    .zw-ctl label{display:block;font-weight:600;font-size:13px;margin:0 0 4px}
    .zw-ctl select{width:100%;min-height:44px;font:inherit;font-size:15px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
    .zw-btn{min-height:44px;padding:8px 14px;border:1px solid var(--ink);border-radius:6px;background:var(--card);color:var(--ink);font:inherit;font-weight:600;cursor:pointer}
    .zw-chk{display:flex;align-items:center;gap:6px;min-height:44px;font-size:14px}
    .zw-step{font-weight:700;margin:4px 0 8px}.zw-mute{color:var(--mute)}.zw-small{font-size:13px}
    .zw-row{display:grid;grid-template-columns:minmax(0,9.5em) minmax(0,1fr) 3.6em;gap:8px;align-items:center;font-size:14px;padding:3px 0}
    .zw-row .zb{height:14px;background:var(--line);border-radius:0 4px 4px 0;position:relative}.zw-row .zb i{position:absolute;inset:0 auto 0 0;background:var(--zw-a);border-radius:0 4px 4px 0}
    .zw-row .zv{text-align:right;font-variant-numeric:tabular-nums}.zw-row .zn{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .zw-teams{font-size:12.5px;color:var(--mute);margin:-2px 0 6px 0;line-height:1.35}
    .zw-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--mute);margin:6px 0 0}
    .zw-legend span{display:inline-flex;align-items:center;gap:5px}.zw-sw{display:inline-block;width:28px;height:10px;border-radius:2px;background:linear-gradient(90deg,transparent,var(--zw-a))}
    .zw-tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:7px 9px;border-radius:6px;font-size:13px;line-height:1.35;opacity:0;max-width:240px;z-index:50}
  </style>`);
  root.classList.add('zw');
  root.innerHTML = `<div class="zw-ctl">
      <div><label>Map</label><select class="zw-mapsel">${MAPS.map(m => `<option>${esc(m)}</option>`).join('')}</select></div>
      <div><label>Ring 1 centred near</label><select class="zw-r1sel"></select></div>
      <div><label>Ring 2 centred near</label><select class="zw-r2sel"></select></div>
      <button class="zw-btn zw-reset" type="button">Clear rings</button>
      <label class="zw-chk"><input type="checkbox" class="zw-algs"> ALGS games only</label>
      <label class="zw-chk"><input type="checkbox" class="zw-tm" checked> Show teams' landing spots</label>
    </div>
    <div class="zw-grid"><div><div class="zw-map"><svg viewBox="0 0 1 1" role="img" aria-label="Map of where final zones end"></svg></div>
      <div class="zw-legend"><span><i class="zw-sw"></i>where the final zone ended (darker = more often)</span><span>dashed = rings you placed</span><span><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="var(--zw-hot)"/></svg> a Champs team's usual landing spot</span></div></div>
      <div><p class="zw-step"></p><div class="zw-list"></div><p class="zw-small zw-mute zw-foot"></p></div></div>`;
  const $ = s => root.querySelector(s), svg = $('svg');
  const tip = document.body.appendChild(Object.assign(document.createElement('div'), {className: 'zw-tip'}));
  const hov = (el, html) => { el.addEventListener('mousemove', e => { tip.innerHTML = html(); tip.style.opacity = 1; tip.style.left = Math.min(innerWidth - 250, e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; }); el.addEventListener('mouseleave', () => tip.style.opacity = 0); };

  function fillSel() {
    const d = data(S.map), names = d.m.outlines.map(o => o[0]).sort();
    $('.zw-r1sel').innerHTML = `<option value="">– click the map, or pick –</option>` + names.map(n => `<option>${esc(n)}</option>`).join('');
    $('.zw-r2sel').innerHTML = `<option value="">– after ring 1 –</option>` + names.map(n => `<option>${esc(n)}</option>`).join('');
  }
  function place(pt) {
    const d = data(S.map), R = d.m.radii;
    if (!S.r1 || S.r2) { S.r1 = pt; S.r2 = null; }
    else {
      const dist = Math.hypot(pt[0] - S.r1[0], pt[1] - S.r1[1]), lim = R[0] - R[1];
      S.r2 = dist <= lim ? pt : [S.r1[0] + (pt[0] - S.r1[0]) * lim / dist, S.r1[1] + (pt[1] - S.r1[1]) * lim / dist];
    }
    draw();
  }
  function draw() {
    const d = data(S.map), m = d.m, R = m.radii, w = weights(d);
    // heat grid
    const G = 48, grid = new Float64Array(G * G); let tot = 0, tot2 = 0;
    const spotW = {};
    for (let k = 0; k < d.n; k++) { const wk = w[k]; if (wk < 1e-6) continue; tot += wk; tot2 += wk * wk;
      const gx = Math.min(G - 1, Math.floor(d.e[2 * k] * G)), gy = Math.min(G - 1, Math.floor(d.e[2 * k + 1] * G));
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) { const x = gx + a, y = gy + b; if (x >= 0 && y >= 0 && x < G && y < G) grid[y * G + x] += wk * (a || b ? (a && b ? 0.25 : 0.5) : 1); }
      const s = d.spot[k]; if (s) spotW[s] = (spotW[s] || 0) + wk; }
    const gm = Math.max(...grid) || 1, cell = 1 / G, effN = tot2 ? tot * tot / tot2 : 0;
    let h = `<defs><filter id="zw-blur-${uid}" x="-5%" y="-5%" width="110%" height="110%"><feGaussianBlur stdDeviation=".005"/></filter></defs><rect width="1" height="1" fill="var(--card)"/>`;
    h += m.outlines.map(([n, p]) => `<polygon points="${p.map(q => q.join(',')).join(' ')}" fill="var(--line)" fill-opacity=".55" stroke="var(--mute)" stroke-opacity=".5" stroke-width=".002"/>`).join('');
    h += `<g filter="url(#zw-blur-${uid})">`;
    if (effN >= 8) for (let i = 0; i < G * G; i++) if (grid[i] > gm * 0.04) h += `<rect x="${(i % G) * cell}" y="${Math.floor(i / G) * cell}" width="${cell}" height="${cell}" fill="var(--zw-a)" fill-opacity="${(0.1 + 0.8 * Math.sqrt(grid[i] / gm)).toFixed(2)}"/>`;
    h += `</g>`;
    if (S.r1) h += `<circle cx="${S.r1[0]}" cy="${S.r1[1]}" r="${R[0]}" fill="none" stroke="var(--ink)" stroke-width=".004" stroke-dasharray=".015 .01"/><circle cx="${S.r1[0]}" cy="${S.r1[1]}" r=".006" fill="var(--ink)"/>`;
    if (S.r2) h += `<circle cx="${S.r2[0]}" cy="${S.r2[1]}" r="${R[1]}" fill="none" stroke="var(--ink)" stroke-width=".005" stroke-dasharray=".01 .007"/><circle cx="${S.r2[0]}" cy="${S.r2[1]}" r=".006" fill="var(--ink)"/>`;
    const tm = {};
    if (S.teams && opt.teams) Object.entries(opt.teams).forEach(([t, mp]) => { const s = mp[S.map]; if (s && d.cent[s]) (tm[s] = tm[s] || []).push(t); });
    Object.entries(tm).forEach(([s, ts]) => { const [x, y] = d.cent[s]; h += `<circle class="zw-team" data-s="${esc(s)}" cx="${x}" cy="${y + 0.022}" r="${0.008 + 0.002 * ts.length}" fill="var(--zw-hot)" stroke="var(--card)" stroke-width=".003"/>`; });
    h += m.outlines.map(([n]) => { const [x, y] = d.cent[n]; return `<text x="${x}" y="${y}" font-size=".022" text-anchor="middle" fill="var(--ink)" stroke="var(--card)" stroke-width=".006" paint-order="stroke" style="pointer-events:none;font-weight:600">${esc(n)}</text>`; }).join('');
    svg.innerHTML = h;
    svg.querySelectorAll('.zw-team').forEach(c => hov(c, () => `<b>${esc(c.dataset.s)}</b><br>Usual spot of ${tm[c.dataset.s].map(esc).join(', ')}`));
    // list
    const eff = tot2 ? tot * tot / tot2 : 0, list = Object.entries(spotW).map(([s, v]) => [s, v / tot]).filter(r => r[1] >= 0.005).sort((a, b) => b[1] - a[1]).slice(0, 7), mx = list.length ? list[0][1] : 1;
    const ck = m.check;
    $('.zw-step').textContent = !S.r1 ? `Where the final zone has ended on ${S.map} (${S.algs ? 'ALGS games' : Math.round(eff).toLocaleString() + ' games'}). Click the map to place ring 1.`
      : !S.r2 ? `Ring 1 placed. Likeliest final spots from about ${Math.round(eff)} past games with a similar ring 1. Click again to place ring 2.`
      : `Ring 2 placed. Likeliest final spots from about ${Math.round(eff)} past games that started like this. Click anywhere to start over.`;
    $('.zw-list').innerHTML = eff < 8 ? '<p class="zw-mute">Too few past games match these rings. Try moving a ring, or turn off "ALGS games only".</p>' :
      list.map(([s, p]) => `<div class="zw-row"><span class="zn">${esc(s)}</span><span class="zb"><i style="width:${(100 * p / mx).toFixed(1)}%"></i></span><span class="zv">${pct(p)}</span></div>${tm[s] ? `<div class="zw-teams">Lands here: ${tm[s].map(esc).join(', ')}</div>` : ''}`).join('');
    $('.zw-foot').innerHTML = `How sure? Tested on past games it hadn't seen: after ring 1 the top spot was right ${pct(ck.r1_top1)} of the time and the top three ${pct(ck.r1_top3)}; after ring 2, ${pct(ck.top1)} and ${pct(ck.top3)}. From map history alone it would be ${pct(ck.top1_base)} and ${pct(ck.top3_base)}. Ring data: ${esc(ZD.source)} (${d.n.toLocaleString()} recorded games on current ring settings, ${(ZD.maps[S.map].algs ? ZD.maps[S.map].algs.split(',').length : 0)} of them ALGS). Landing-spot outlines: apexlegendsstatus.com.`;
    $('.zw-r1sel').value = ''; $('.zw-r2sel').value = '';
    if (opt.onRings) opt.onRings({map: S.map, r1: S.r1, r2: S.r2, cent: d.cent});
  }
  svg.addEventListener('click', e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; const q = p.matrixTransform(svg.getScreenCTM().inverse()); place([Math.max(0, Math.min(1, q.x)), Math.max(0, Math.min(1, q.y))]); });
  $('.zw-mapsel').onchange = e => { S.map = e.target.value; S.r1 = S.r2 = null; fillSel(); draw(); };
  $('.zw-r1sel').onchange = e => { const c = data(S.map).cent[e.target.value]; if (c) { S.r1 = c; S.r2 = null; draw(); } };
  $('.zw-r2sel').onchange = e => { const c = data(S.map).cent[e.target.value]; if (c && S.r1) { S.r2 = null; place(c); } else if (c) { S.r1 = c; draw(); } };
  $('.zw-reset').onclick = () => { S.r1 = S.r2 = null; draw(); };
  $('.zw-algs').onchange = e => { S.algs = e.target.checked; draw(); };
  $('.zw-tm').onchange = e => { S.teams = e.target.checked; draw(); };
  fillSel(); draw();
  return {setMap: m => { if (ZD.maps[m]) { S.map = m; $('.zw-mapsel').value = m; S.r1 = S.r2 = null; fillSel(); draw(); } }};
}
