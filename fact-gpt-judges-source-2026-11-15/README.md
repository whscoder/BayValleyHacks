# Fact GPT

A Chrome extension that identifies opinion and factual framing, highlights bias,
and checks selected factual claims against independent web sources.

## What judges should look at

- **Working application:** `front-end/` (Chrome extension) and `back-end/home.py` (FastAPI).
- **ML approach and honest results:** [docs/ML_RESULTS.md](docs/ML_RESULTS.md).
- **System flow and file guide:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Verification:** 110 backend tests and six frontend test files are included.
- **Deployment:** a Dockerfile and pinned Python requirements are included.

This is a curated source repository, not the full research workspace. It includes
one production LR artifact, not downloaded transformer weights, raw datasets,
credentials, experiment caches, or old extension releases. Codex assisted with
implementation, data audits, debugging, documentation, and automated tests.

## Try the extension

1. Open `chrome://extensions` and enable **Developer mode**.
2. Click **Load unpacked** and select this repository's `front-end/` folder.
3. Pin **Fact GPT**, open a public news article, and click Analyze.

The extension currently targets `https://bias-article-detector.onrender.com`.
That server is a separate deployment; uploading this repository does not update it.
Its availability, API budget and allowed extension IDs are controlled by the owner.
If your newly loaded extension ID is not allowed, use the local setup below.
No API key belongs in the extension.

## Run your own backend

Use Python 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env
```

Edit `.env`: supply your own OpenAI API key and a private random signing secret.
Copy the extension ID from `chrome://extensions` into `FACTGPT_ALLOWED_ORIGINS`
as `chrome-extension://YOUR_EXTENSION_ID`. Your account needs access to the API
models configured in `back-end/home.py`. Actual article analysis makes paid API
requests; the offline tests below do not.

For local extension requests, change `configuredHostedUrl` in
`front-end/config.js` to `http://127.0.0.1:8000` and add
`http://127.0.0.1:8000/*` to `host_permissions` in `front-end/manifest.json`.
Reload the extension after those edits, then start the backend:

```bash
python -m uvicorn home:app --app-dir back-end --env-file .env --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`. Windows users can activate their environment
with `.venv\Scripts\activate` instead. FFmpeg is needed only for the optional
podcast endpoints, not the article demo. The Dockerfile already installs it.

### Docker alternative

After preparing `.env` and your extension's backend URL:

```bash
docker build -t fact-gpt .
docker run --rm --env-file .env -p 8000:8000 fact-gpt
```

Never add `.env` to Git or configure a secret token inside the public extension.
For a hosted deployment, set secrets in your hosting provider and allow the exact
extension origin(s) that will use that deployment.

## Run the offline checks

After activating the Python environment, run from the repository root:

```bash
PYTHONPATH=back-end python -m unittest -q test_fact_opinion_route test_article_jobs test_home_quotes test_research_contract test_podcast
node --test front-end/test_*.mjs
```

Node.js 20+ is sufficient for these optional frontend checks. Tests mock the
provider requests; a real OpenAI key is not required to run them. Passing these
checks does not establish live provider availability or research accuracy.

## Scope and limitations

The current extension demo is Article mode. Shared podcast utilities remain in
the source because the backend and frontend import them; podcast mode is not
presented as a completed extension feature. Research checks up to three selected
claims, not every assertion. Factual-type classification is not proof of truth.
Research uses Terra with low reasoning, starts alongside bias analysis after
classification, prefers relevant .gov/.edu/.org evidence, and rejects the input
article as its own corroboration. Unsupported claims remain unclear.

## Upload to your new repository

Upload the **contents of this folder** to the new repository root, not the ZIP
itself and not the original research workspace. Include hidden files such as
`.gitignore` and `.env.example`. The included production pickle must remain at
`data/fact_opinion/processed/fact_opinion_classifier.pkl`.
