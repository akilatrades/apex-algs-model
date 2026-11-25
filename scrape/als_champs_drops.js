// Run on apexlegendsstatus.com. Latest landing spot per map for every team, newest Y6 Split 2 event first
// (LCQ finals, Playoffs, Pro League finals and days 9-7). Read window.__drop afterwards; keep Championship teams -> data/deep/champs_drops.txt
window.__drop = null;
(async () => {
  const names = {};
  for (const m of ['mp_rr_tropic_island_mu2','mp_rr_district','mp_rr_desertlands_mu5','mp_rr_olympus_mu3']) { (await fetch('/tournament/local-api/landingGetZones?map='+m).then(r=>r.json())).forEach(a => names[a.id] = a.name); }
  const MAPS = {mp_rr_tropic_island_mu2: 'SP', mp_rr_olympus_mu3: 'OL', mp_rr_district: 'ED', mp_rr_desertlands_mu5: 'WE'};
  const U = ['LCQ/NA/Finals/Finals','LCQ/EMEA/Finals/Finals','LCQ/APAC_N/Finals/Finals','LCQ/APAC_S/Finals/Finals',
    'ALGS-Playoffs/Global/Finals/Finals','ALGS-Playoffs/Global/Day3/LosersRound2','ALGS-Playoffs/Global/Day3/WinnersRound1','ALGS-Playoffs/Global/Day3/LosersRound1',
    'ALGS-Playoffs/Global/Day2/AvD','ALGS-Playoffs/Global/Day2/BvC','ALGS-Playoffs/Global/Day2/AvC','ALGS-Playoffs/Global/Day1/BvD','ALGS-Playoffs/Global/Day1/CvD','ALGS-Playoffs/Global/Day1/AvB',
    'Pro-League/NA/Finals/Finals','Pro-League/EMEA/Finals/Finals','Pro-League/APAC_N/Finals/Finals','Pro-League/APAC_S/Finals/Finals',
    'Pro-League/NA/Day9/BvC','Pro-League/EMEA/Day9/BvC','Pro-League/APAC_N/Day9/BvC','Pro-League/APAC_S/Day9/BvC',
    'Pro-League/NA/Day8/AvC','Pro-League/EMEA/Day8/AvC','Pro-League/APAC_N/Day8/AvC','Pro-League/APAC_S/Day8/AvC',
    'Pro-League/NA/Day7/AvB','Pro-League/EMEA/Day7/AvB','Pro-League/APAC_N/Day7/AvB','Pro-League/APAC_S/Day7/AvB'];
  const got = {}, out = [];
  for (const u of U) {
    const h = await fetch('/algs/Y6-Split2/' + u).then(r => r.text());
    const m = h.match(/let dropsData = (\{.*?\});\s*\n/s); if (!m) continue;
    for (const [mp, teams] of Object.entries(JSON.parse(m[1]))) for (const t of Object.values(teams)) {
      const k = t.teamName.trim() + '|' + (MAPS[mp] || mp);
      if (t.poi != null && !got[k]) { got[k] = 1; out.push(k + '|' + (names[t.poi] || t.poi)); }
    }
  }
  window.__drop = out.join('\n');
})();
'started'
