# Synthetic data annotator

## 1. Google Sheet (once)

1. Create a Google Sheet, then open **Extensions → Apps Script**.
2. Replace the default code with `anotatzailea_apps_script.gs` and save.
3. Go to **Project Settings (⚙) → Script properties → Add**, set `PASSWORD` to your chosen password, and save.
4. Go to **Deploy → New deployment → Web app**. Set **Execute as:** Me and **Who has access:** Anyone. Deploy, authorize, and copy the `/exec` URL.

## 2. Build the page

```bash
cd openrouter/annotation
python3 build_anotatzailea.py -n 10 -seed 0 -p 1.0 -i ../output -o anotatzailea.html -u "https://script.google.com/macros/s/XXXX/exec"
```

Required: `-n` items per model, `-seed` seed. `-p` sets the share of synthetic pairs vs. correct-only sentences (default `0.8` → 80/20; if one kind runs out, the other fills the gap). Optional paths: `-i` outputs folder, `-o` HTML, `-k` key (`eredu_gakoa.json`), `-t` template (`-h` lists the defaults).

- This creates `anotatzailea.html` from `openrouter/output/`. Rebuild whenever new outputs arrive.
- Leave out `-u` for a local-only page: annotators send their files instead (step 4).
- Paste the `id <tab> model` lines the build prints into the Sheet's **`gakoa`** tab (`id` in A1, `eredua` in B1).

## 3. Send it

- Send **only** `anotatzailea.html`, and tell the annotators the password.
- **Never** share `eredu_gakoa.json` (hidden ID → model name). Keep it: if it's lost, the IDs change.

## 4. Results

- **Sheet:** `Laburpena` has scores per model; `anotazioak` holds every annotated item in a single tab, with `sarrera` (file) and `eredua` (model id) columns to filter by; raw data is in the hidden `raw_data` sheet (do not delete it: the web app reads and writes there).
- **Local only:** annotators click **⬇ Gorde fitxategia** and send you `sintetikoak_*.json`.
- Score per item: **1** if *Zuzena* **and** *Lotuta*, otherwise **0**.

## Notes

- Always use the same `-n`, `-seed` and `-p`: changing them changes which sentences are shown.
- To change the password, edit `PASSWORD`. There's no need to redeploy.
- After editing the script, use **Manage deployments → Edit → New version**, which keeps the same URL.
- Files the build needs: `build_anotatzailea.py`, `anotatzailea_template.html`, and `output/` (`.json` + `.meta.json`).
