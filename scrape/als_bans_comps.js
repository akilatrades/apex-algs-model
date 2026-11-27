// Run on apexlegendsstatus.com (built-in browser). For each stage: the ban in each game (from the replay banners) and each
// team's legends in each game (from the per-game page's legends_team). Background job: poll window.__bc_done / window.__bc.
// Output lines: stage|game|id10|map|ban|team=Leg1+Leg2+Leg3;team=...   ban "-" when none ("Would have been banned" = no ban).
// Resumable: progress is kept in localStorage ('bc_lines', 'bc_i'), so a page reload only loses the stage in flight.
window.__bc=null;
(async()=>{
 const STAGES=__STAGES__;
 const jsonAt=(h,name)=>{let i=h.indexOf('var '+name+' = '); if(i<0) return null; i=h.indexOf('{',i); let d=0,q=false,e=false;
   for(let k=i;k<h.length;k++){const c=h[k]; if(q){ if(e) e=false; else if(c==='\\') e=true; else if(c==='"') q=false; continue;}
     if(c==='"') q=true; else if(c==='{') d++; else if(c==='}'){ d--; if(!d) return JSON.parse(h.slice(i,k+1)); } } return null;};
 const out=JSON.parse(localStorage.getItem('bc_lines')||'[]'); let i0=+(localStorage.getItem('bc_i')||0);
 for (let si=i0; si<STAGES.length; si++){ const [code,url]=STAGES[si];
  let h; try{ h=await fetch(url).then(r=>r.text()); }catch(e){ out.push('ERR|'+code); localStorage.setItem('bc_i',si+1); continue; }
  const chunks=h.split('replay-banner_fade').slice(1);
  for (const c of chunks){
   const t=c.match(/Game #(\d+) - ([^<]+)<\/h3>/), id=c.match(/\/algs\/game\/([0-9a-f]{32})/); if(!t||!id) continue;
   const b=c.match(/>Ban: ([^<]+)</); const ban=b? b[1].trim() : '-';
   let comps='';
   try{ const g=await fetch(url+'/'+id[1]).then(r=>r.text()); const lt=jsonAt(g,'legends_team')||{};
     comps=Object.entries(lt).map(([tm,v])=>tm.trim()+'='+Object.keys(v.teamComps||{})[0]?.replace(/\|/g,'+')).join(';'); }catch(e){ comps='ERR'; }
   out.push([code,t[1],id[1].slice(0,10),t[2].trim(),ban,comps].join('|'));
  }
  localStorage.setItem('bc_lines',JSON.stringify(out)); localStorage.setItem('bc_i',si+1);
 }
 window.__bc=out.join('\n');
})(); 'started'
