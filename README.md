# DP Dataset Factory

Automated pipeline for building a large, verified dynamic-programming
reasoning dataset for training and evaluating coding models.

## Pipeline

Raw sources
    ↓
Collection
    ↓
Normalization
    ↓
Deduplication
    ↓
DP classification
    ↓
Taxonomy
    ↓
Solution extraction
    ↓
Verification
    ↓
Synthetic generation
    ↓
Adversarial generation
    ↓
SFT / RL datasets

## Goal

Build a high-quality dataset containing:

- real DP problems
- transformed DP problems
- compositional DP problems
- adversarial DP problems
- correct solutions
- incorrect solutions
- debugging traces
- DP state/transition annotations
- executable verification data

## Commands

```bash
dpfactory collect
dpfactory normalize
dpfactory deduplicate
dpfactory classify
dpfactory taxonomy
dpfactory verify
dpfactory generate
dpfactory verify-generated
dpfactory export
---

# 10. Add a Makefile

```bash
cat > Makefile <<'EOF'
.PHONY: install test lint typecheck collect normalize deduplicate classify taxonomy verify generate export

install:
	pip install -e .

test:
	pytest -q

lint:
	ruff check src tests

typecheck:
	mypy src

collect:
	dpfactory collect

normalize:
	dpfactory normalize

deduplicate:
	dpfactory deduplicate

classify:
	dpfactory classify

taxonomy:
	dpfactory taxonomy

verify:
	dpfactory verify

generate:
	dpfactory generate

export:
	dpfactory export
