#!/usr/bin/env python3
"""Datu sintetikoen anotatzailea: output/-eko JSONak HTML autonomo batean txertatzen ditu.

Erabilera:

    python3 build_anotatzailea.py -n 10 -seed 0 -i ../output -o anotatzailea.html [-u https://script.google.com/macros/s/.../exec]

output/<sarrera>/<eredua>.json bakoitzetik --n adibide sintetiko hautatzen dira (araueen artean
txandaka, hazi finkoarekin: berreraikitzean lagin bera); --pair-ratio-k bikote/esaldi proportzioa
finkatzen du (lehenetsia 0.8 -> %80 bikoteak). Ereduen izenak ez dira HTMLan sartzen:
identifikatzaile opako bat (e1a2b3c) erabiltzen da, eta gakoa --key fitxategian gordetzen da
(ez partekatu anotatzaileekin).
"""
import argparse
import hashlib
import json
import random
import re
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "/*__EMBEDDED_DATA__*/null"
SYNC_MARKER = "/*__SYNC_URL__*/null"


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_key(path: Path):
    """Gakoa: {salt, models: {id: slug}}. Gatza behin sortzen da, id-ak egonkorrak izan daitezen."""
    if path.is_file():
        key = json.loads(path.read_text(encoding="utf-8"))
    else:
        key = {"salt": secrets.token_hex(8), "models": {}}
    return key


def model_id(key, slug: str) -> str:
    mid = "e" + hashlib.sha256((key["salt"] + slug).encode()).hexdigest()[:6]
    key["models"][mid] = slug
    return mid


def round_robin(pools, k: int):
    """Arauen artean txandaka hartu, k item edo poolak agortu arte."""
    out = []
    while len(out) < k and any(pools):
        for pool in pools:
            if pool and len(out) < k:
                out.append(pool.pop())
    return out


def sample_items(rules, n: int, rng: random.Random, pair_ratio: float = 0.8):
    """n item hautatu: ~pair_ratio bikoteak, gainerakoak esaldi zuzenak (arauen artean txandaka).
    Mota batek nahikoa ez badu, besteak betetzen du. JSONeko ordena mantentzen da."""
    pair_pools, correct_pools = [], []
    for ri, r in enumerate(rules):
        pairs, correct = [], []
        for k, p in enumerate(r.get("synthetic_pairs") or []):
            if isinstance(p, dict) and (p.get("original") or p.get("corrected")):
                pairs.append({"iid": f"{ri}p{k}", "kind": "pair", "r": ri, "o": k,
                              "neg": p.get("original", ""), "pos": p.get("corrected", "")})
        n_pairs = len(r.get("synthetic_pairs") or [])
        for k, s in enumerate(r.get("synthetic_correct") or []):
            if isinstance(s, str) and s.strip():
                correct.append({"iid": f"{ri}c{k}", "kind": "correct", "r": ri, "o": n_pairs + k, "pos": s})
        rng.shuffle(pairs)
        rng.shuffle(correct)
        pair_pools.append(pairs)
        correct_pools.append(correct)
    k_pairs = round(n * pair_ratio)
    out = round_robin(pair_pools, k_pairs)
    out += round_robin(correct_pools, n - len(out))
    out += round_robin(pair_pools, n - len(out))  # esaldi zuzenak falta badira, bikoteekin bete
    out.sort(key=lambda it: (it["r"], it.pop("o")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", "--output-dir", dest="output_dir", metavar="DIR",
                    default=str(HERE.parent / "output"), help="compare_models.py-ren irteera (lehenetsia: %(default)s)")
    ap.add_argument("-o", "--out", metavar="FILE.html", default=str(HERE / "anotatzailea.html"),
                    help="sortutako HTMLa (lehenetsia: %(default)s)")
    ap.add_argument("-k", "--key", metavar="FILE.json", default=str(HERE / "eredu_gakoa.json"),
                    help="id opakoa -> eredua, EZ partekatu (lehenetsia: %(default)s)")
    ap.add_argument("-t", "--template", metavar="FILE.html", default=str(HERE / "anotatzailea_template.html"),
                    help="HTML txantiloia (lehenetsia: %(default)s)")
    ap.add_argument("-n", "--n", type=int, required=True, help="eredu eta sarrera bakoitzeko item kopurua (adib. 10)")
    ap.add_argument("-p", "--pair-ratio", type=float, required=True,
                    help="bikoteen proportzioa laginean, 0-1 (adib. 0.8 -> 80/20 bikote/esaldi)")
    ap.add_argument("-seed", "--seed", dest="seed", required=True, help="laginketaren hazia (adib. 0)")
    ap.add_argument("-u", "--sync-url", metavar="URL", default=None, help="Google Apps Script web app-aren URLa (/exec)")
    args = ap.parse_args()
    if not 0.0 <= args.pair_ratio <= 1.0:
        ap.error("--pair-ratio 0 eta 1 artean egon behar da")

    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        print(f"ERROR: ez da existitzen {out_dir}", file=sys.stderr)
        return 1

    key_path = Path(args.key)
    key = load_key(key_path)
    prompt = None
    entries = []
    for d in sorted((p for p in out_dir.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        models, grammar = [], None
        for f in sorted(d.glob("*.json")):
            if f.name.endswith(".meta.json"):
                continue
            meta_path = f.with_name(f.stem + ".meta.json")
            try:
                rules = json.loads(f.read_text(encoding="utf-8"))
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            except json.JSONDecodeError as e:
                print(f"  AVISO: {f} ezin da irakurri ({e})")
                continue
            if not isinstance(rules, list):
                print(f"  AVISO: {f} ez da zerrenda bat, saltatzen")
                continue
            slug = meta.get("model") or f.stem
            if grammar is None and meta.get("grammar_text"):
                grammar = meta["grammar_text"]
                if prompt is None and meta.get("prompt"):
                    prompt = meta["prompt"].replace(grammar, "{grammar_text}")
            rng = random.Random(f"{args.seed}|{d.name}|{slug}")
            items = sample_items(rules, args.n, rng, args.pair_ratio)
            # arauen testua ez da txertatzen: anotatzaileak ez du ikusi behar
            models.append({"id": model_id(key, slug), "items": items})
            n_pair = sum(it["kind"] == "pair" for it in items)
            print(f"  {d.name} / {slug}: {len(rules)} arau, {len(items)} item ({n_pair} bikote, {len(items) - n_pair} esaldi)")
        if models:
            entries.append({"name": d.name, "grammar": grammar or "", "models": models})

    if not entries:
        print("ERROR: ez da emaitzarik aurkitu", file=sys.stderr)
        return 1

    payload = json.dumps({"prompt": prompt or "", "entries": entries}, ensure_ascii=False).replace("</", "<\\/")
    html = Path(args.template).read_text(encoding="utf-8")
    if MARKER not in html:
        print("ERROR: txantiloiak ez du datu-markatzailerik", file=sys.stderr)
        return 1
    html = html.replace(MARKER, payload, 1)
    if args.sync_url:
        if not args.sync_url.startswith("https://script.google.com/"):
            print("AVISO: URLa ez da https://script.google.com/-ekin hasten; egiaztatu /exec-en amaitzen dela.")
        html = html.replace(SYNC_MARKER, json.dumps(args.sync_url), 1)
    Path(args.out).write_text(html, encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK -> {args.out}  ({len(entries)} sarrera)"
          + ("  (Google Sheets sinkronizazioa aktibatuta)" if args.sync_url else ""))
    print(f"Ereduen gakoa -> {key_path}  (ez partekatu anotatzaileekin)")
    print("Google Sheet-eko 'gakoa' orrian itsasteko (id <tab> eredua):")
    for mid, slug in sorted(key["models"].items(), key=lambda kv: kv[1]):
        print(f"{mid}\t{slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
