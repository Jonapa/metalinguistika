/**
 * Datu sintetikoen anotatzailea <-> Google Sheets.
 *
 * Instalazioa: ikus README.md.
 *
 * Pasahitza: Project Settings > Script properties > PASSWORD. Edonork ireki dezake web app-a,
 * baina pasahitz zuzenik gabe ez da ezer idazten ez irakurtzen. PASSWORD ezarri gabe, dena ukatzen da.
 *
 * Orriak:
 *   raw_data          datu gordinak (ezkutuan): errenkada bat (anotatzaile, sarrera, eredu, item) bakoitzeko
 *   anotazioak        orri bakarra: sarrera eta eredua zutabeetan, itemak JSONeko ordenan, anotatzaile bakoitzeko zutabeak
 *   Laburpena         puntuazioa ereduka, eredu x anotatzaile eta sarrera x eredu
 *   gakoa             id opakoa -> eredu-izena (eskuz bete eredu_gakoa.json-etik; Laburpenak erabiltzen du)
 *
 * Puntuazioa: item bakoitzak (esaldi bakarra edo pare minimoa) puntu 1 du baldin eta
 * correct_ok = 1 ETA rule_ok = 1. Puntuazioa = puntuak / bi baldintzak anotatuta dituzten itemak.
 */
const RAW = 'raw_data', ITEMS = 'anotazioak', SUMMARY = 'Laburpena', KEYSHEET = 'gakoa';
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

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Anotazioak')
    .addItem('Orri guztiak berreraiki', 'rebuildAll').addToUi();
}

function sheet_(name, header) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(name);
  // Aurreko bertsioek datu gordinak 'anotazioak' orrian gordetzen zituzten: berrizendatu, ez gainidazteko
  if (!sh && name === RAW) {
    const old = ss.getSheetByName(ITEMS);
    if (old && old.getRange(1, 1).getValue() === COLS[0]) sh = old.setName(RAW);
  }
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
    const fresh = [];
    Object.keys(latest).forEach(k => {
      const r = latest[k];
      const vals = COLS.map(c => c === 'updated' ? now : (r[c] == null ? '' : r[c]));
      if (index[k]) sh.getRange(index[k], 1, 1, COLS.length).setValues([vals]);
      else fresh.push(vals);
    });
    if (fresh.length) sh.getRange(sh.getLastRow() + 1, 1, fresh.length, COLS.length).setValues(fresh);
    rebuildAll();
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
  writeItemsSheet_(rows);
  writeSummary_(rows);
  const raw = sheet_(RAW, COLS);                         // datu gordinak: beharrezkoak, baina ezkutuan
  if (!raw.isSheetHidden()) raw.hideSheet();
}

// (sarrera, eredu) bakoitzeko bloke bat: itemak, GUZTIRA eta oharrak; sarrera/eredua zutabeen bidez iragaz daiteke
function writeItemsSheet_(rows) {
  const annotators = [...new Set(rows.map(r => String(r[C.annotator])))].sort();
  const header = ['sarrera', 'eredua', 'item', 'mota', '✗ negatiboa', '✓ positiboa'];
  annotators.forEach(a => header.push(`${a} · zuzena`, `${a} · araua`, `${a} · puntua`));
  const out = [header], blocks = [];                     // blocks: {start, items, total, end} errenkada-zenbakiak (1etik)
  const groups = [...new Set(rows.map(r => r[C.entry] + '|' + r[C.model]))].sort();
  groups.forEach(k => {
    const [entry, model] = k.split('|');
    const items = {}, notes = [];
    rows.filter(r => String(r[C.entry]) === entry && String(r[C.model]) === model).forEach(r => {
      if (r[C.item] === '_oharrak') { if (String(r[C.observations]).trim()) notes.push([r[C.annotator], r[C.observations]]); return; }
      const it = items[r[C.item]] = items[r[C.item]] || {row: r, by: {}};
      it.by[r[C.annotator]] = r;
    });
    // A01-E01, A01-E02, … A02-E01: araua + adibidea, zeroz beteta testu gisa ordenatzean ere ordena bera izateko
    const ids = Object.keys(items).sort(cmpItems_), perRule = {}, labels = {};
    ids.forEach(id => { const ri = itemOrder_(id)[0] + 1; labels[id] = [ri, perRule[ri] = (perRule[ri] || 0) + 1]; });
    const w = v => String(Math.max(...Object.values(labels).map(l => l[v]), 10)).length;
    const wr = w(0), we = w(1), pad0 = (v, n) => String(v).padStart(n, '0');
    const block = {start: out.length + 1, items: ids.length, total: null};
    ids.forEach(id => {
      const r = items[id].row, [ri, ei] = labels[id];
      const line = [entry, model, `A${pad0(ri, wr)}-E${pad0(ei, we)}`, r[C.kind] === 'pair' ? 'pare minimoa' : 'esaldia', r[C.negative], r[C.positive]];
      annotators.forEach(a => {
        const x = items[id].by[a];
        line.push(x ? x[C.correct_ok] : '', x ? x[C.rule_ok] : '', x ? x[C.score] : '');
      });
      out.push(line);
    });
    if (ids.length) {
      const total = [entry, model, 'GUZTIRA', '', '', 'puntuak / itemak · %'];
      annotators.forEach(a => {
        const sc = Object.values(items).map(it => it.by[a] && num_(it.by[a][C.score])).filter(v => v != null);
        const pts = sc.reduce((s, v) => s + v, 0);
        total.push('', sc.length ? `${pts} / ${sc.length}` : '', sc.length ? Math.round(1000 * pts / sc.length) / 10 : '');
      });
      block.total = out.length + 1;
      out.push(total);
    }
    notes.forEach(([a, t]) => { const l = Array(header.length).fill(''); l[0] = entry; l[1] = model; l[2] = 'Oharrak'; l[3] = a; l[4] = t; out.push(l); });
    block.end = out.length;
    if (block.end >= block.start) blocks.push(block);
  });

  const sh = sheet_(ITEMS);
  if (sh.getFilter()) sh.getFilter().remove();
  sh.clear();
  sh.getRange(1, 1, out.length, header.length).setValues(out);
  formatItemsSheet_(sh, out.length, header.length, annotators.length, blocks);
}

// Itxura: goiburu iluna, (sarrera, eredu) blokeak txandakako koloreetan, ✓/✗ koloreztatuta, GUZTIRA % gradientez
function formatItemsSheet_(sh, nRows, nCols, nAnn, blocks) {
  const FIXED = 6, A1 = (r, c, nr, nc) => sh.getRange(r, c, nr, nc).getA1Notation();
  const BAND = ['#ffffff', '#f3f6fb'], TOTAL_BG = '#e3e9f4', LINE = '#c5cdd9';
  const all = sh.getRange(1, 1, nRows, nCols);
  all.setFontFamily('Roboto').setFontSize(10).setVerticalAlignment('middle');
  sh.setHiddenGridlines(true);

  sh.getRange(1, 1, 1, nCols).setBackground('#2f3e55').setFontColor('#ffffff').setFontWeight('bold')
    .setHorizontalAlignment('center').setWrap(true);
  sh.setRowHeight(1, 42);
  sh.setFrozenRows(1);
  sh.setFrozenColumns(3);

  [150, 90, 85, 105, 330, 330].forEach((w, i) => sh.setColumnWidth(i + 1, w));
  if (nAnn) sh.setColumnWidths(FIXED + 1, 3 * nAnn, 78);
  if (nRows > 1) {
    sh.getRange(2, 5, nRows - 1, 2).setWrap(true);
    sh.getRange(2, 1, nRows - 1, 4).setFontColor('#4a5568');
    if (nAnn) sh.getRange(2, FIXED + 1, nRows - 1, 3 * nAnn).setHorizontalAlignment('center');
  }

  const items = [], totals = [], totalPct = [], notes = [];
  blocks.forEach((b, i) => {
    sh.getRange(b.start, 1, b.end - b.start + 1, nCols).setBackground(BAND[i % 2])
      .setBorder(null, null, true, null, null, null, LINE, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    if (b.items && nAnn) items.push(A1(b.start, FIXED + 1, b.items, 3 * nAnn));
    if (b.total) {
      totals.push(A1(b.total, 1, 1, nCols));
      for (let a = 0; a < nAnn; a++) totalPct.push(A1(b.total, FIXED + 3 * a + 3, 1, 1));
    }
    const firstNote = (b.total || b.start + b.items - 1) + 1;
    if (b.end >= firstNote) notes.push(A1(firstNote, 1, b.end - firstNote + 1, nCols));
  });
  if (totals.length) sh.getRangeList(totals).setBackground(TOTAL_BG).setFontWeight('bold').setFontColor('#1f2a3c');
  if (totalPct.length) sh.getRangeList(totalPct).setNumberFormat('0.0"%"');
  if (notes.length) sh.getRangeList(notes).setFontStyle('italic').setFontColor('#6b7280').setWrap(true);
  if (items.length) sh.getRangeList(items).setNumberFormat('"✓";"−";"✗"');

  // Anotatzaile bakoitzaren hiru zutabeak bereizteko marra bertikala
  for (let a = 0; a < nAnn; a++) sh.getRange(1, FIXED + 3 * a + 1, nRows, 1)
    .setBorder(null, true, null, null, null, null, LINE, SpreadsheetApp.BorderStyle.SOLID);

  const rules = [];
  const itemRanges = items.map(r => sh.getRange(r));
  if (itemRanges.length) {
    rules.push(SpreadsheetApp.newConditionalFormatRule().whenNumberEqualTo(1)
      .setBackground('#d4edda').setFontColor('#1e7b34').setRanges(itemRanges).build());
    rules.push(SpreadsheetApp.newConditionalFormatRule().whenNumberEqualTo(0)
      .setBackground('#f8d7da').setFontColor('#a71d2a').setRanges(itemRanges).build());
  }
  if (totalPct.length) rules.push(SpreadsheetApp.newConditionalFormatRule()
    .setGradientMinpointWithValue('#f4a6a6', SpreadsheetApp.InterpolationType.NUMBER, '0')
    .setGradientMidpointWithValue('#fff3b0', SpreadsheetApp.InterpolationType.NUMBER, '50')
    .setGradientMaxpointWithValue('#8fd19e', SpreadsheetApp.InterpolationType.NUMBER, '100')
    .setRanges(totalPct.map(r => sh.getRange(r))).build());
  sh.setConditionalFormatRules(rules);

  all.createFilter();                                    // sarrera / eredua zutabeen arabera iragazteko
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
  const W = Math.max(6, 1 + entries.length, 1 + annotators.length);
  const pad = l => l.concat(Array(W - l.length).fill(''));
  const out = [], sections = [];                         // sections: {title, head, n, cols, pct: [lehen zutabea, kopurua]}
  const section = (t, head, lines, pct) => {
    sections.push({title: out.length + 1, head: out.length + 2, n: lines.length, cols: head.length, pct});
    out.push(pad([t]), pad(head));
    lines.forEach(l => out.push(pad(l)));
    out.push(pad([]));
  };

  section('1 · Puntuazioa ereduka (anotatzaile eta sarrera guztiak)',
    ['eredua', 'itemak', 'puntuak', 'puntuazioa %', 'zuzena %', 'arauari lotuta %'],
    models.map(m => [label(m)].concat(agg(scored.filter(r => r[C.model] === m))))
      .sort((a, b) => (b[3] || 0) - (a[3] || 0)), [4, 3]);
  section('2 · Puntuazioa % · eredua × anotatzailea', ['eredua'].concat(annotators),
    models.map(m => [label(m)].concat(annotators.map(a =>
      agg(scored.filter(r => r[C.model] === m && r[C.annotator] === a))[2]))), [2, annotators.length]);
  section('3 · Puntuazioa % · eredua × sarrera', ['eredua'].concat(entries),
    models.map(m => [label(m)].concat(entries.map(en =>
      agg(scored.filter(r => r[C.model] === m && r[C.entry] === en))[2]))), [2, entries.length]);
  out.push(pad(['Puntu 1 = zuzena (esaldia, edo pare minimoa osorik) ETA arauari lotuta. Bi baldintzak anotatuta dituzten itemak soilik.']));

  const sh = sheet_(SUMMARY);
  sh.clear();
  sh.getRange(1, 1, out.length, W).setValues(out);
  formatSummary_(sh, out.length, W, sections);
}

// Itxura: atal-izenburu handiak, goiburu ilunak, errenkada txandakatuak eta % zutabeak gradientez (0 gorria → 100 berdea)
function formatSummary_(sh, nRows, W, sections) {
  const LINE = '#c5cdd9', BAND = ['#ffffff', '#f3f6fb'];
  sh.getRange(1, 1, nRows, W).setFontFamily('Roboto').setFontSize(10).setVerticalAlignment('middle');
  sh.setHiddenGridlines(true);
  sh.setColumnWidth(1, 280);
  if (W > 1) sh.setColumnWidths(2, W - 1, 105);
  const pctRanges = [];
  sections.forEach(s => {
    sh.getRange(s.title, 1).setFontSize(13).setFontWeight('bold').setFontColor('#2f3e55');
    sh.setRowHeight(s.title, 34);
    sh.getRange(s.title, 1, 1, s.cols).setVerticalAlignment('bottom')
      .setBorder(null, null, true, null, null, null, '#2f3e55', SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    sh.getRange(s.head, 1, 1, s.cols).setBackground('#2f3e55').setFontColor('#ffffff').setFontWeight('bold')
      .setWrap(true).setHorizontalAlignment('center');
    sh.getRange(s.head, 1).setHorizontalAlignment('left');
    sh.setRowHeight(s.head, 36);
    if (!s.n) return;
    for (let i = 0; i < s.n; i++) sh.getRange(s.head + 1 + i, 1, 1, s.cols).setBackground(BAND[i % 2]);
    const body = sh.getRange(s.head + 1, 1, s.n, s.cols);
    body.setBorder(null, null, true, null, null, true, LINE, SpreadsheetApp.BorderStyle.SOLID);
    sh.getRange(s.head + 1, 1, s.n, 1).setFontWeight('bold').setFontColor('#1f2a3c');
    if (s.cols > 1) sh.getRange(s.head + 1, 2, s.n, s.cols - 1).setHorizontalAlignment('center');
    if (s.pct[1]) {
      const r = sh.getRange(s.head + 1, s.pct[0], s.n, s.pct[1]);
      r.setNumberFormat('0.0"%"');
      pctRanges.push(r);
    }
  });
  const foot = sh.getRange(nRows, 1);
  foot.setFontStyle('italic').setFontColor('#6b7280');
  if (pctRanges.length) sh.setConditionalFormatRules([SpreadsheetApp.newConditionalFormatRule()
    .setGradientMinpointWithValue('#f4a6a6', SpreadsheetApp.InterpolationType.NUMBER, '0')
    .setGradientMidpointWithValue('#fff3b0', SpreadsheetApp.InterpolationType.NUMBER, '50')
    .setGradientMaxpointWithValue('#8fd19e', SpreadsheetApp.InterpolationType.NUMBER, '100')
    .setRanges(pctRanges).build()]);
  else sh.setConditionalFormatRules([]);
}
