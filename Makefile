.PHONY: check figures report test all
all: check test figures report
check:    ; python analysis/recompute_reported_numbers.py
test:     ; python tests/test_bootstrap.py
figures:  ; python analysis/make_figures.py
report:   ; cd report && latexmk -xelatex -interaction=nonstopmode main.tex
