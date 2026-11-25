// Run on an eternalesports.club /lobbies/{id}/{tag} page. Returns JSON with games + per-team rows.
(() => {
  const main = document.querySelector('main');
  const L = main.innerText.split('\n').map(s => s.trim()).filter(Boolean);
  const leaves = new Set([...main.querySelectorAll('*')].filter(e => e.childElementCount == 0)
    .map(e => e.textContent.trim()).filter(Boolean));
  const games = [];
  for (let i = 0; i < L.length; i++) {
    const m = L[i].match(/^Game (\d+)$/);
    if (m && L[i + 2] && /•/.test(L[i + 2])) games.push({ game: +m[1], map: L[i + 1], when: L[i + 2] });
  }
  const splitName = s => {
    for (let i = s.length - 1; i > 0; i--) {
      const a = s.slice(0, i), b = s.slice(i);
      if (leaves.has(a) && leaves.has(b)) return a;
    }
    return s;
  };
  const rows = [];
  const n = games.length;
  for (let i = 0; i < L.length; i++) {
    if (L[i] !== 'KILLS' || L[i + 2] !== 'DAMAGE') continue;
    let j = i - 1; const nameLine = L[j];
    let k = j - 1; let group = null;
    if (/^[A-H]$/.test(L[k])) { group = L[k]; k--; }
    const total = +L[k], rank = +L[k - 1];
    const kills = +L[i + 1], damage = +L[i + 3].replace(/,/g, '');
    let q = i + 6; const pl = [], pts = [];
    for (let g = 0; g < n; g++) {
      const t = L[q];
      if (/^\d+(st|nd|rd|th)$/.test(t)) { pl.push(parseInt(t)); pts.push(+L[q + 1]); q += 2; }
      else { pl.push(null); pts.push(null); q += /^-$/.test(t) ? 1 : 0; }
    }
    rows.push({ rank, total, group, team: splitName(nameLine), raw: nameLine, kills, damage, placements: pl, points: pts });
  }
  const title = L[0], date = L[1];
  const key = location.pathname.split('/').slice(2,4).join('-');
  return JSON.stringify({ key, url: location.pathname, title, date, games, rows });
})()
