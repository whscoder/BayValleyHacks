# Architecture and file guide

```text
Chrome article -> background job -> FastAPI extraction -> local LR + guarded API review
                                                       |                 |
                                                       v                 v
                                                   bias analysis     source research
                                                       |                 |
                                                       +-> validated result -> popup/highlights
```

After classification, article bias and research run concurrently. Pending research
is cancelled/awaited if the owning analysis fails. Closing the popup does not
cancel a server-owned article job; reopening can resume polling.

- `front-end/manifest.json`, `extension.html`, `extension.css`: extension entry/UI.
- `front-end/popup.js`, `popupRenderState.js`, `background.js`: UI and job lifecycle.
- `front-end/article.js`, `parseScript.js`: article helpers and validated rendering.
- `front-end/api.js`, `backendClient.js`, `config.js`: backend requests/configuration.
- `front-end/podcast.js`: shared imported helpers; not a completed Article-mode UI feature.
- `back-end/home.py`: endpoints, extraction, LR review policy, orchestration and validation.
- `back-end/ai_prompts.py`: bias/research instructions and structured schemas.
- `back-end/podcast.py`: imported transcript/media utilities for optional endpoints.
- `back-end/Ml_datapreprocessing.py`, `mltraintest.py`: optional data/training source.
- `data/fact_opinion/processed/fact_opinion_classifier.pkl`: supplied production LR pipeline.
- `back-end/test_*.py`, `front-end/test_*.mjs`: mocked/offline runtime checks.
- `docs/results/`: small aggregate evidence, not full corpora or private article logs.

The local model performs binary fact/opinion classification. Clear opinions can
resolve locally at >=0.90 confidence, respecting stricter artifact thresholds;
possible factual clauses, attribution, forecasts and other risk cues still receive
API review. Factual thresholds remain artifact-derived. OpenAI can return mixed
passages with exact opinion excerpts. Review requests are batched.

Research receives the original URL and independently selects factual claims.
A citation must appear in completed web-search provenance and cannot be the input
article or its recognized tracking/mobile/AMP alias. Relevant .gov/.edu/.org pages
are prioritized, but the page must support the specific claim. Research schema,
quote offsets, classification signatures and browser origin/URL checks are enforced.

The supplied model is a trusted sklearn pickle. Do not load arbitrary pickles from
untrusted sources or replace it with an experimental artifact bundle without checking
serialization, vectorizer and serving compatibility. Jobs and caches are process-local;
restarting the backend loses in-flight jobs. Docker supplies Chromium dependencies.
