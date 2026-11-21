# Seminar slides

- `slides.tex` (+ `temp_slide.tex`) — Beamer deck, Latin Modern fonts, styled after the department's reference deck. Build: `pdflatex slides.tex` twice.
- `make_slide_figs.py` — draws every chart in `figs/` from `data/`. Run from the repo root.
- `temp_analysis.py` — temperature follow-up: full-history repair rerun at T = 0.8 (`experiment2/temp_experiment.py`, `experiment2/run_temp.ps1`, data in `data/temperature/`).
- `live_demo.py` — live on-stage demo with a GPU energy meter. Copy into the experiment folder next to `harness.py`.
- `speaker_notes.md` — what to say on each click.
