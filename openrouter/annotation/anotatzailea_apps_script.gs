/**
 * Datu sintetikoen anotatzailea <-> Google Sheets.
 *
 * Instalazioa: ikus README.md.
 *
 * Pasahitza: Project Settings > Script properties > PASSWORD. Edonork ireki dezake web app-a,
 * baina pasahitz zuzenik gabe ez da ezer idazten ez irakurtzen. PASSWORD ezarri gabe, dena ukatzen da.
 *
 * Orriak:
 *   anotazioak        datu gordinak (ezkutuan): errenkada bat (anotatzaile, sarrera, eredu, item) bakoitzeko
 *   <sarrera> · <id>  JSON (sarrera x eredu) bakoitzeko orri bat: itemak JSONeko ordenan, anotatzaile bakoitzeko zutabeak
 *   Laburpena         puntuazioa ereduka, eredu x anotatzaile eta sarrera x eredu
 *   gakoa             id opakoa -> eredu-izena (eskuz bete eredu_gakoa.json-etik; Laburpenak erabiltzen du)
 *
 * Puntuazioa: item bakoitzak (esaldi bakarra edo pare minimoa) puntu 1 du baldin eta
 * correct_ok = 1 ETA rule_ok = 1. Puntuazioa = puntuak / bi baldintzak anotatuta dituzten itemak.
 */
const RAW = 'anotazioak', SUMMARY = 'Laburpena', KEYSHEET = 'gakoa';
const COLS = ['annotator', 'entry', 'model', 'position', 'item', 'kind', 'rule_index',
              'negative', 'positive', 'correct_ok', 'rule_ok', 'score', 'observations', 'updated'];
const C = Object.fromEntries(COLS.map((c, i) => [c, i]));
// Konstante-denborako konparaketa, pasahitza karakterez karaktere ez asmatzeko
function passwordOk_(given) {
  const real = PropertiesService.getScriptProperties().getProperty('PASSWORD');
  if (!real || typeof given !== 'string' || given.length !== real.length) return false;
  let diff = 0;
  for (let i = 0; i < real.length; i++) diff |= real.charCodeAt(i) ^ given.charCodeAt(i);
  return diff === 0;
}
const KEY = r => [r.annotator, r.entry, r.model, r.item].join('|');
const jsonSheetName = (entry, model) => `${entry} · ${model}`;

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Anotazioak')
    .addItem('Orri guztiak berreraiki', 'rebuildAll').addToUi();
}

function sheet_(name, header) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(name);
  if (!sh) {
    sh = ss.insertSheet(name);
    if (header) { sh.appendRow(header); sh.setFrozenRows(1); }
  }
  return sh;
}

function json_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}

function readRaw_() {
  const sh = sheet_(RAW, COLS), n = sh.getLastRow() - 1;
  return n > 0 ? sh.getRange(2, 1, n, COLS.length).getValues() : [];
}

const num_ = v => (v === '' || v == null) ? null : Number(v);
// "1p3" -> [1, 0, 3]: araua, mota (pareak lehenago, JSONean bezala), posizioa
const itemOrder_ = it => { const m = String(it).match(/^(\d+)([pc])(\d+)$/); return m ? [+m[1], m[2] === 'p' ? 0 : 1, +m[3]] : [1e9, 0, 0]; };
const cmpItems_ = (a, b) => { const x = itemOrder_(a), y = itemOrder_(b); return x[0] - y[0] || x[1] - y[1] || x[2] - y[2]; };

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const body = JSON.parse(e.postData.contents);
    if (!body || !passwordOk_(body.password)) return json_({ok: false, error: 'auth'});
    const recs = Array.isArray(body.records) ? body.records : [];
    const sh = sheet_(RAW, COLS);
    const index = {};
    readRaw_().forEach((row, i) => {
      index[KEY({annotator: row[C.annotator], entry: row[C.entry], model: row[C.model], item: row[C.item]})] = i + 2;
    });
    const now = new Date();
    const latest = {};                                   // gako bera bi aldiz badator, azkenak irabazten du
    recs.forEach(r => { if (r && r.annotator && r.entry && r.model && r.item) latest[KEY(r)] = r; });
    const fresh = [], touched = {};
    Object.keys(latest).forEach(k => {
      const r = latest[k];
      const vals = COLS.map(c => c === 'updated' ? now : (r[c] == null ? '' : r[c]));
      if (index[k]) sh.getRange(index[k], 1, 1, COLS.length).setValues([vals]);
      else fresh.push(vals);
      touched[r.entry + '|' + r.model] = true;
    });
    if (fresh.length) sh.getRange(sh.getLastRow() + 1, 1, fresh.length, COLS.length).setValues(fresh);
    rebuild_(Object.keys(touched));
    return json_({ok: true, n: recs.length});
  } catch (err) {
    return json_({ok: false, error: String(err)});
  } finally {
    lock.releaseLock();
  }
}

function doGet(e) {
  const p = (e && e.parameter) || {};
  if (p.action !== 'records') return json_({ok: true});
  if (!passwordOk_(p.password)) return json_({ok: false, error: 'auth'});
  const who = String(p.annotator || '').trim();
  const records = readRaw_()
    .filter(row => String(row[C.annotator]).trim() === who)
    .map(row => Object.fromEntries(COLS.map((c, i) => [c, row[i] === '' ? null : row[i]])));
  return json_({ok: true, records});
}

/* ============ ORRI ERATORRIAK ============ */
function rebuildAll() {
  const rows = readRaw_();
  rebuild_([...new Set(rows.map(r => r[C.entry] + '|' + r[C.model]))], rows);
}

function rebuild_(jsonKeys, rows) {
  rows = rows || readRaw_();
  jsonKeys.forEach(k => { const [entry, model] = k.split('|'); writeJsonSheet_(entry, model, rows); });
  writeSummary_(rows);
  const raw = sheet_(RAW, COLS);                         // datu gordinak: beharrezkoak, baina ezkutuan
  if (!raw.isSheetHidden()) raw.hideSheet();
}

function writeJsonSheet_(entry, model, rows) {
  rows = rows.filter(r => r[C.entry] === entry && r[C.model] === model);
  const annotators = [...new Set(rows.map(r => String(r[C.annotator])))].sort();
  const items = {}, notes = [];
  rows.forEach(r => {
    if (r[C.item] === '_oharrak') { if (String(r[C.observations]).trim()) notes.push([r[C.annotator], r[C.observations]]); return; }
    const it = items[r[C.item]] = items[r[C.item]] || {row: r, by: {}};
    it.by[r[C.annotator]] = r;
  });
  const header = ['item', 'mota', '✗ negatiboa', '✓ positiboa'];
  annotators.forEach(a => header.push(`${a} · zuzena`, `${a} · araua`, `${a} · puntua`));
  const out = [header];
  // A01-E01, A01-E02, … A02-E01: araua + adibidea, zeroz beteta testu gisa ordenatzean ere ordena bera izateko
  const ids = Object.keys(items).sort(cmpItems_), perRule = {}, labels = {};
  ids.forEach(id => { const ri = itemOrder_(id)[0] + 1; labels[id] = [ri, perRule[ri] = (perRule[ri] || 0) + 1]; });
  const w = v => String(Math.max(...Object.values(labels).map(l => l[v]), 10)).length;
  const wr = w(0), we = w(1), pad0 = (v, n) => String(v).padStart(n, '0');
  ids.forEach(id => {
    const r = items[id].row, [ri, ei] = labels[id];
    const line = [`A${pad0(ri, wr)}-E${pad0(ei, we)}`, r[C.kind] === 'pair' ? 'pare minimoa' : 'esaldia', r[C.negative], r[C.positive]];
    annotators.forEach(a => {
      const x = items[id].by[a];
      line.push(x ? x[C.correct_ok] : '', x ? x[C.rule_ok] : '', x ? x[C.score] : '');
    });
    out.push(line);
  });
  const total = ['GUZTIRA', '', '', 'puntuak / itemak · %'];
  annotators.forEach(a => {
    const sc = Object.values(items).map(it => it.by[a] && num_(it.by[a][C.score])).filter(v => v != null);
    const pts = sc.reduce((s, v) => s + v, 0);
    total.push('', `${pts} / ${sc.length}`, sc.length ? Math.round(1000 * pts / sc.length) / 10 : '');
  });
  out.push(total);
  notes.forEach(([a, t]) => { const l = Array(header.length).fill(''); l[0] = 'Oharrak'; l[1] = a; l[2] = t; out.push(l); });

  const sh = sheet_(jsonSheetName(entry, model));
  sh.clearContents();
  sh.getRange(1, 1, out.length, header.length).setValues(out);
  sh.setFrozenRows(1);
  sh.getRange(1, 1, 1, header.length).setFontWeight('bold');
  sh.getRange(out.length - notes.length, 1, 1, header.length).setFontWeight('bold');
}

function modelNames_() {
  const sh = sheet_(KEYSHEET, ['id', 'eredua']), n = sh.getLastRow() - 1;
  const names = {};
  if (n > 0) sh.getRange(2, 1, n, 2).getValues().forEach(([id, name]) => { if (id) names[id] = name; });
  return {sh, names};
}

function writeSummary_(rows) {
  const {sh: keySh, names} = modelNames_();
  const scored = rows.filter(r => r[C.item] !== '_oharrak' && num_(r[C.score]) != null);
  const models = [...new Set(rows.map(r => String(r[C.model])))].sort();
  const missing = models.filter(m => !(m in names));
  if (missing.length) keySh.getRange(keySh.getLastRow() + 1, 1, missing.length, 2).setValues(missing.map(m => [m, '']));
  const label = m => names[m] ? `${names[m]} (${m})` : m;
  const agg = list => {
    const n = list.length, pts = list.reduce((s, r) => s + num_(r[C.score]), 0);
    const ok = list.reduce((s, r) => s + num_(r[C.correct_ok]), 0), rule = list.reduce((s, r) => s + num_(r[C.rule_ok]), 0);
    const pct = v => n ? Math.round(1000 * v / n) / 10 : '';
    return [n, pts, pct(pts), pct(ok), pct(rule)];
  };
  const annotators = [...new Set(scored.map(r => String(r[C.annotator])))].sort();
  const entries = [...new Set(scored.map(r => String(r[C.entry])))].sort();
  const W = Math.max(6, 1 + entries.length, 2 + annotators.length);
  const pad = l => l.concat(Array(W - l.length).fill(''));
  const out = [], bold = [];
  const title = t => { bold.push(out.length + 1); out.push(pad([t])); };

  title('1. Puntuazioa ereduka (anotatzaile eta sarrera guztiak)');
  bold.push(out.length + 1);
  out.push(pad(['eredua', 'itemak', 'puntuak', 'puntuazioa %', 'zuzena %', 'arauari lotuta %']));
  models.map(m => [label(m)].concat(agg(scored.filter(r => r[C.model] === m))))
    .sort((a, b) => (b[3] || 0) - (a[3] || 0)).forEach(l => out.push(pad(l)));
  out.push(pad([]));

  title('2. Puntuazioa % · eredua x anotatzailea');
  bold.push(out.length + 1);
  out.push(pad(['eredua'].concat(annotators)));
  models.forEach(m => out.push(pad([label(m)].concat(annotators.map(a =>
    agg(scored.filter(r => r[C.model] === m && r[C.annotator] === a))[2])))));
  out.push(pad([]));

  title('3. Puntuazioa % · eredua x sarrera');
  bold.push(out.length + 1);
  out.push(pad(['eredua'].concat(entries)));
  models.forEach(m => out.push(pad([label(m)].concat(entries.map(en =>
    agg(scored.filter(r => r[C.model] === m && r[C.entry] === en))[2])))));
  out.push(pad([]));
  out.push(pad(['Puntu 1 = zuzena (esaldia, edo pare minimoa osorik) ETA arauari lotuta. Bi baldintzak anotatuta dituzten itemak soilik.']));

  const sh = sheet_(SUMMARY);
  sh.clearContents();
  sh.getRange(1, 1, out.length, W).setValues(out);
  sh.getRange(1, 1, out.length, W).setFontWeight('normal');
  bold.forEach(r => sh.getRange(r, 1, 1, W).setFontWeight('bold'));
}
