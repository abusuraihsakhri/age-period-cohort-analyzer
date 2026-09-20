"use strict";

const $ = (id) => document.getElementById(id);
const EPS = 1e-12;
const SAMPLE = `age_group,period,events,person_years
40-44,2000-2004,34,210000
40-44,2005-2009,30,220000
40-44,2010-2014,26,230000
40-44,2015-2019,22,240000
45-49,2000-2004,78,205000
45-49,2005-2009,70,215000
45-49,2010-2014,61,225000
45-49,2015-2019,52,235000
50-54,2000-2004,152,198000
50-54,2005-2009,141,208000
50-54,2010-2014,126,218000
50-54,2015-2019,109,228000
55-59,2000-2004,248,190000
55-59,2005-2009,236,200000
55-59,2010-2014,214,210000
55-59,2015-2019,189,220000`;

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("apc-theme", theme);
}
setTheme(localStorage.getItem("apc-theme") || "light");
$("themeToggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === button));
    document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === button.dataset.tab));
  });
});

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field.trim()); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field.trim()); field = "";
      if (row.some((x) => x !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  row.push(field.trim());
  if (row.some((x) => x !== "")) rows.push(row);
  if (quoted) throw new Error("CSV contains an unterminated quoted field.");
  if (rows.length < 2) throw new Error("CSV must include a header and at least one data row.");
  const headers = rows[0].map((x) => x.toLowerCase());
  const required = ["age_group", "period", "events", "person_years"];
  required.forEach((h) => { if (!headers.includes(h)) throw new Error(`Missing required column: ${h}`); });
  return rows.slice(1).map((values, index) => {
    const obj = {};
    headers.forEach((h, i) => obj[h] = values[i] ?? "");
    obj.events = Number(obj.events);
    obj.person_years = Number(obj.person_years);
    if (!Number.isFinite(obj.events) || obj.events < 0) throw new Error(`Row ${index + 2}: events must be non-negative.`);
    if (!Number.isFinite(obj.person_years) || obj.person_years <= 0) throw new Error(`Row ${index + 2}: person_years must be positive.`);
    if (!obj.age_group || !obj.period) throw new Error(`Row ${index + 2}: age_group and period cannot be blank.`);
    return obj;
  });
}

function buildTable(records) {
  const ages = [...new Set(records.map((r) => r.age_group))];
  const periods = [...new Set(records.map((r) => r.period))];
  const keyMap = new Map(records.map((r) => [`${r.age_group}\u0000${r.period}`, r]));
  if (keyMap.size !== records.length) throw new Error("Duplicate age_group / period cells are not allowed.");
  if (records.length !== ages.length * periods.length) throw new Error("Records must form a complete rectangular age × period table.");
  const cells = [];
  ages.forEach((age, ai) => periods.forEach((period, pi) => {
    const rec = keyMap.get(`${age}\u0000${period}`);
    if (!rec) throw new Error(`Missing cell: ${age} / ${period}`);
    cells.push({ ageIdx: ai, periodIdx: pi, cohortIdx: pi - ai + ages.length - 1, events: rec.events, py: rec.person_years });
  }));
  return { ages, periods, cohorts: ages.length + periods.length - 1, cells };
}

function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[pivot][c])) pivot = r;
    if (Math.abs(M[pivot][c]) < 1e-12) throw new Error("Model design is singular for this dataset.");
    [M[c], M[pivot]] = [M[pivot], M[c]];
    const pv = M[c][c];
    for (let j = c; j <= n; j++) M[c][j] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row) => row[n]);
}

function invert(A) {
  const n = A.length;
  const cols = Array.from({ length: n }, (_, i) => {
    const e = Array(n).fill(0); e[i] = 1; return solve(A, e);
  });
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => cols[j][i]));
}

function logGamma(z) {
  const p = [0.9999999999998099,676.5203681218851,-1259.1392167224028,771.3234287776531,-176.6150291621406,12.5073432786869,-0.13857109526572,9.984369578019572e-6,1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1; let x = p[0];
  for (let i = 1; i < p.length; i++) x += p[i] / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function glm(X, y, exposure) {
  const n = y.length, p = X[0].length;
  if (n < p) throw new Error("This table is too small for the requested model.");
  const beta = Array(p).fill(0);
  beta[0] = Math.log(y.reduce((a,b)=>a+b,0) / exposure.reduce((a,b)=>a+b,0));
  const off = exposure.map(Math.log);
  let cov = null, mu = null;
  const logLik = (b) => X.reduce((acc, row, i) => {
    const eta = off[i] + row.reduce((s, v, j) => s + v * b[j], 0);
    const m = Math.exp(Math.max(-700, Math.min(700, eta)));
    return acc + y[i] * eta - m - logGamma(y[i] + 1);
  }, 0);

  for (let iter = 0; iter < 100; iter++) {
    const eta = X.map((row, i) => off[i] + row.reduce((s, v, j) => s + v * beta[j], 0));
    mu = eta.map((v) => Math.exp(Math.max(-700, Math.min(700, v))));
    const score = Array(p).fill(0), info = Array.from({length:p},()=>Array(p).fill(0));
    X.forEach((row, i) => {
      for (let j = 0; j < p; j++) {
        score[j] += row[j] * (y[i] - mu[i]);
        for (let k = j; k < p; k++) info[j][k] += row[j] * mu[i] * row[k];
      }
    });
    for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) info[j][k] = info[k][j];
    const scale = Math.max(1, ...info.map((r, i) => r[i]));
    for (let j = 0; j < p; j++) info[j][j] += scale * 1e-12;
    const delta = solve(info, score), old = logLik(beta);
    let step = 1, candidate = beta.slice(), accepted = false;
    while (step >= 1e-6) {
      candidate = beta.map((v, j) => v + step * delta[j]);
      if (logLik(candidate) >= old - 1e-10) { accepted = true; break; }
      step *= 0.5;
    }
    if (!accepted) throw new Error("Poisson model failed to converge.");
    candidate.forEach((v, j) => beta[j] = v);
    if (Math.max(...delta.map((v) => Math.abs(step * v))) < 1e-9) break;
    if (iter === 99) throw new Error("Poisson model did not converge.");
  }
  const eta = X.map((row, i) => off[i] + row.reduce((s, v, j) => s + v * beta[j], 0));
  mu = eta.map((v) => Math.exp(Math.max(-700, Math.min(700, v))));
  const info = Array.from({length:p},()=>Array(p).fill(0));
  X.forEach((row, i) => { for (let j = 0; j < p; j++) for (let k = j; k < p; k++) info[j][k] += row[j] * mu[i] * row[k]; });
  for (let j=0;j<p;j++) for (let k=0;k<j;k++) info[j][k]=info[k][j];
  const scale = Math.max(1, ...info.map((r,i)=>r[i]));
  for (let j=0;j<p;j++) info[j][j]+=scale*1e-12;
  cov = invert(info);
  let dev = 0;
  y.forEach((obs, i) => { dev += obs > 0 ? 2 * (obs * Math.log(obs / mu[i]) - (obs - mu[i])) : 2 * mu[i]; });
  return { beta, cov, mu, logLik: logLik(beta), deviance: Math.max(0, dev), k: p };
}

function gammaQ(a, x) {
  if (x === 0) return 1;
  const gln = logGamma(a);
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let i = 0; i < 1000; i++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-14) break; }
    return Math.max(0, Math.min(1, 1 - sum * Math.exp(-x + a * Math.log(x) - gln)));
  }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-14) break;
  }
  return Math.max(0, Math.min(1, Math.exp(-x + a * Math.log(x) - gln) * h));
}

function modelDesign(table, kind) {
  const A = table.ages.length, P = table.periods.length, C = table.cohorts;
  return table.cells.map((c) => {
    const row = [1];
    for (let a = 1; a < A; a++) row.push(c.ageIdx === a ? 1 : 0);
    if (kind === "AP" || kind === "APC") for (let p = 1; p < P; p++) row.push(c.periodIdx === p ? 1 : 0);
    if (kind === "AC") for (let co = 1; co < C; co++) row.push(c.cohortIdx === co ? 1 : 0);
    else if (kind === "APC") for (let co = 1; co < C - 1; co++) row.push(c.cohortIdx === co ? 1 : 0);
    return row;
  });
}

function analyzeTable(table) {
  const y = table.cells.map((c) => c.events), exp = table.cells.map((c) => c.py), n = y.length;
  const specs = [["Age-Only (A)","A"],["Age-Period (AP)","AP"],["Age-Cohort (AC)","AC"],["Age-Period-Cohort (APC)","APC"]];
  const models = specs.map(([name, kind]) => {
    const fit = glm(modelDesign(table, kind), y, exp), df = n - fit.k;
    return { name, deviance: fit.deviance, df, aic: -2*fit.logLik + 2*fit.k, bic: -2*fit.logLik + Math.log(n)*fit.k, p: df > 0 ? gammaQ(df/2, fit.deviance/2) : NaN };
  });
  const A = table.ages.length, P = table.periods.length, center = (P - 1) / 2, interval = inferPeriodInterval(table.periods);
  const X = table.cells.map((c) => [1, ...Array.from({length:A-1},(_,i)=>c.ageIdx===i+1?1:0), (c.periodIdx-center)*interval]);
  const adjusted = glm(X, y, exp), slope = adjusted.beta[adjusted.beta.length-1], se = Math.sqrt(Math.max(0, adjusted.cov.at(-1).at(-1)));
  const net = (Math.exp(slope)-1)*100, lo=(Math.exp(slope-1.96*se)-1)*100, hi=(Math.exp(slope+1.96*se)-1)*100;
  const local = {};
  table.ages.forEach((age, ai) => {
    const cells = table.cells.filter((c)=>c.ageIdx===ai).sort((a,b)=>a.periodIdx-b.periodIdx);
    const xa = cells.map((c)=>[1,(c.periodIdx-center)*interval]);
    const f = glm(xa, cells.map((c)=>c.events), cells.map((c)=>c.py));
    local[age]=(Math.exp(f.beta[1])-1)*100;
  });
  return { models, net, lo, hi, local, lowest: models.reduce((a,b)=>a.aic < b.aic ? a : b) };
}

function inferPeriodInterval(labels) {
  const mids = labels.map((label) => {
    const nums = String(label).match(/\d{4}/g);
    if (!nums) return null;
    const n = nums.map(Number); return n.length >= 2 ? (n[0]+n[1])/2 : n[0];
  });
  const diffs = [];
  for (let i=1;i<mids.length;i++) if (mids[i] !== null && mids[i-1] !== null && mids[i] > mids[i-1]) diffs.push(mids[i]-mids[i-1]);
  return diffs.length ? diffs.sort((a,b)=>a-b)[Math.floor(diffs.length/2)] : 1;
}

function updateShape() {
  try {
    const records = parseCsv($("csvInput").value), table = buildTable(records);
    $("dataShape").textContent = `${table.ages.length} ages × ${table.periods.length} periods · ${records.length} cells`;
  } catch { $("dataShape").textContent = "Dataset not yet valid"; }
}

function renderAnalysis(table, result) {
  $("emptyState").classList.add("hidden"); $("errorBox").classList.add("hidden"); $("results").classList.remove("hidden");
  $("netDrift").textContent = `${result.net >= 0 ? "+" : ""}${result.net.toFixed(2)}%/yr`;
  $("netDriftCi").textContent = `95% CI ${result.lo.toFixed(2)} to ${result.hi.toFixed(2)}`;
  $("lowestAic").textContent = result.lowest.name.replace("Age-", "").replace(" (", " · ").replace(")", "");
  $("tableSize").textContent = `${table.ages.length} × ${table.periods.length}`;
  $("cohortCount").textContent = `${table.cohorts} cohort diagonals`;
  $("modelRows").innerHTML = result.models.map((m) => `<tr class="${m===result.lowest?'lowest-row':''}"><td>${m.name}</td><td>${m.deviance.toFixed(3)}</td><td>${m.df}</td><td>${m.aic.toFixed(2)}</td><td>${m.bic.toFixed(2)}</td><td>${Number.isFinite(m.p)?m.p.toPrecision(3):'n/a'}</td></tr>`).join("");
  $("localDrifts").innerHTML = Object.entries(result.local).map(([age,v])=>`<span class="chip">${age}<b>${v>=0?'+':''}${v.toFixed(2)}%</b></span>`).join("");
}

function showError(message) {
  $("results").classList.add("hidden"); $("emptyState").classList.add("hidden"); $("errorBox").textContent = message; $("errorBox").classList.remove("hidden");
}

$("loadSample").addEventListener("click", () => { $("csvInput").value = SAMPLE; updateShape(); });
$("csvInput").addEventListener("input", updateShape);
$("fileInput").addEventListener("change", async (e) => { const file=e.target.files[0]; if(file){ $("csvInput").value=await file.text(); updateShape(); } });
$("analyzeButton").addEventListener("click", () => { try { const table=buildTable(parseCsv($("csvInput").value)); renderAnalysis(table, analyzeTable(table)); } catch(e) { showError(e.message || String(e)); } });

function calcPaf() {
  const p=Number($("pafPrev").value), rr=Number($("pafRr").value);
  if(!Number.isFinite(p)||p<0||p>1||!Number.isFinite(rr)||rr<1){ $("pafResult").textContent="Check inputs"; return; }
  const value=p*(rr-1)/(1+p*(rr-1)); $("pafResult").textContent=`${(value*100).toFixed(2)}%`;
}
$("pafButton").addEventListener("click", calcPaf); calcPaf();

function ols(xs, ys) {
  const n=xs.length, mx=xs.reduce((a,b)=>a+b,0)/n, my=ys.reduce((a,b)=>a+b,0)/n;
  const sxx=xs.reduce((s,x)=>s+(x-mx)**2,0); if(sxx<=EPS) throw new Error("Years must contain distinct values.");
  const slope=xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0)/sxx, intercept=my-slope*mx;
  const res=xs.map((x,i)=>ys[i]-(intercept+slope*x)), sse=res.reduce((s,r)=>s+r*r,0), mse=n>2?sse/(n-2):0;
  return {slope,intercept,se:Math.sqrt(Math.max(0,mse/sxx)),mse,mx,sxx,sse};
}

function segmentedTrend(years,rates,maxJ=2,minLen=4){
  const n=years.length, logs=rates.map(Math.log), candidates=[];
  const add=(bounds)=>{ const starts=[0,...bounds], stops=[...bounds,n], segs=[]; let sse=0;
    starts.forEach((s,i)=>{const e=stops[i], f=ols(years.slice(s,e),logs.slice(s,e)); sse+=f.sse; segs.push({s,e,fit:f});});
    const k=2*segs.length, bic=n*Math.log(Math.max(sse/n,1e-15))+k*Math.log(n); candidates.push({bounds,segs,sse,bic}); };
  add([]); if(maxJ>=1&&n>=2*minLen) for(let s=minLen;s<=n-minLen;s++) add([s]);
  if(maxJ>=2&&n>=3*minLen) for(let s1=minLen;s1<=n-2*minLen;s1++) for(let s2=s1+minLen;s2<=n-minLen;s2++) add([s1,s2]);
  const best=candidates.reduce((a,b)=>a.bic<b.bic?a:b); let totalW=0, sum=0; best.segs.forEach(seg=>{const w=Math.max(1,seg.e-seg.s-1);totalW+=w;sum+=seg.fit.slope*w;});
  return {aapc:(Math.exp(sum/totalW)-1)*100, joinpoints:best.bounds.map(i=>years[i])};
}

function forecast(years,rates,horizon){
  const f=ols(years,rates.map(Math.log)), out=[]; const last=years.at(-1), n=years.length;
  for(let h=1;h<=horizon;h++){const year=last+h, pred=f.intercept+f.slope*year, se=Math.sqrt(Math.max(0,f.mse*(1+1/n+(year-f.mx)**2/f.sxx)));out.push({year,mid:Math.exp(pred),lo:Math.exp(pred-1.96*se),hi:Math.exp(pred+1.96*se)});} return out;
}

$("trendButton").addEventListener("click",()=>{
  try{
    const years=$("trendYears").value.split(",").map(x=>Number(x.trim())), rates=$("trendRates").value.split(",").map(x=>Number(x.trim()));
    if(years.length!==rates.length||years.length<4) throw new Error("Use the same number of years and rates (at least 4).");
    if(years.some((v,i)=>!Number.isFinite(v)||(i&&v<=years[i-1]))) throw new Error("Years must be finite and strictly increasing.");
    if(rates.some(v=>!Number.isFinite(v)||v<=0)) throw new Error("Rates must be positive.");
    const maxJ=Number($("maxJoinpoints").value), t=segmentedTrend(years,rates,maxJ,Math.min(4,Math.max(2,Math.floor(years.length/(maxJ+1))))), h=Number($("forecastHorizon").value), fc=forecast(years,rates,h);
    const boxes=$("trendResult").querySelectorAll("strong"); boxes[0].textContent=`${t.aapc>=0?'+':''}${t.aapc.toFixed(2)}%/yr`; boxes[1].textContent=t.joinpoints.length?t.joinpoints.join(", "):"None"; boxes[2].textContent=`${fc.at(-1).year}: ${fc.at(-1).mid.toFixed(2)} (${fc.at(-1).lo.toFixed(2)}–${fc.at(-1).hi.toFixed(2)})`;
  }catch(e){const boxes=$("trendResult").querySelectorAll("strong"); boxes[0].textContent="Error"; boxes[1].textContent=e.message; boxes[2].textContent="—";}
});

$("csvInput").value = SAMPLE;
updateShape();
