// Run on https://placement.gg (any page). Builds data/rings/ring_sequences.txt content in window.__pg.
// Uses placement.gg's own map transform: u = 0.5 + (x + ox) / (2 size), v = 0.5 - (y + oy) / (2 sizeY)  (same frame as the
// apexlegendsstatus.com landing-spot outlines; checked visually against the map art on all four maps, 25 Nov 2026).
// Keeps the current ring profile, seasons 28+. Record = r1 u,v r2 u,v (1 char each, /63) + endzone u,v (9 bits each, 3 chars).
window.__pg = null;
(async () => {
  const B = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_', out = [];
  const cl = (v, n) => Math.max(0, Math.min(n, Math.round(v * n)));
  for (const code of ['mp_rr_district_mu1', 'mp_rr_tropic_island_mu2_landscape', 'mp_rr_desertlands_mu5', 'mp_rr_olympus_mu3']) {
    const d = await fetch(`/data/zones/${code}.json?v=dev`).then(r => r.json());
    const c = {}; d.pid.forEach((p, i) => { if (+d.sn[i] >= 28) c[p] = (c[p] || 0) + 1; });
    const P = +Object.entries(c).sort((a, b) => b[1] - a[1])[0][0];
    const U = x => 0.5 + (x + d.ox) / (2 * d.size), V = y => 0.5 - (y + d.oy) / (2 * d.sizeY);
    const rows = [], algs = [];
    for (let i = 0; i < d.n; i++) {
      if (d.pid[i] !== P || +d.sn[i] < 28) continue;
      const xy = d.xy.slice(i * 10, i * 10 + 10), f = k => k % 2 === 0 ? U(xy[k]) : V(xy[k]);
      let s = ''; for (const k of [0, 1, 2, 3]) s += B[cl(f(k), 63)];
      const v = cl(f(8), 511) * 512 + cl(f(9), 511); s += B[v >> 12] + B[(v >> 6) & 63] + B[v & 63];
      if ((d.evt[i] || []).length) algs.push(rows.length); rows.push(s);
    }
    out.push(['#' + d.name, d.profiles[P].radii.map(r => (r / (2 * d.size)).toFixed(5)).join(','), rows.length].join('|'), rows.join(''), 'A:' + algs.join(','));
  }
  window.__pg = out.join('\n');
})();
'started'
