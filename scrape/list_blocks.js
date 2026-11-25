// Run on any eternalesports.club page: lists played scrim blocks for the season.
(async () => { const r = await fetch('/api/lobby?seasonId=5684').then(r => r.json());
  return JSON.stringify({ played: r.keys, upcoming: (r.lobbyInstances || []).map(x => x.id + '-' + x.tag + ' ' + x.name) }); })()
