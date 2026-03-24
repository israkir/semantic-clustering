# Run `make` or `make help` for grouped, colorized targets.
# Quick start: make cluster-viz
# Variables: PROMPTS_FILE, SHOW=1 / INTERACTIVE=1

REPO_ROOT := $(abspath .)
OUTPUT_DIR := $(REPO_ROOT)/outputs
PYTHON_DIR := $(REPO_ROOT)/python
VENV_DIR := $(PYTHON_DIR)/.venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_STAMP := $(VENV_DIR)/.install-stamp
PROMPTS_FILE ?= $(REPO_ROOT)/data/prompts.txt
SHOW ?= 0
INTERACTIVE ?= $(SHOW)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OPEN_CMD := open
else
	OPEN_CMD := xdg-open
endif

_esc := $(shell printf '\033')
BOLD   := $(_esc)[1m
DIM    := $(_esc)[2m
RED    := $(_esc)[31m
GREEN  := $(_esc)[32m
YELLOW := $(_esc)[33m
BLUE   := $(_esc)[34m
MAG    := $(_esc)[35m
CYAN   := $(_esc)[36m
WHITE  := $(_esc)[37m
RESET  := $(_esc)[0m

.DEFAULT_GOAL := help

.PHONY: help cluster-viz outputs-dir java-viz python-viz clean clean-data clean-venv

help:
	@printf '$(BOLD)$(CYAN)%s$(RESET)\n' 'semantic-clustering'
	@printf '$(DIM)%s$(RESET)\n' '  Cluster user prompts: Tribuo HDBSCAN (Java) · Sentence-Transformers + UMAP + HDBSCAN (Python)'
	@printf '\n'
	@printf '$(BOLD)$(MAG)%s$(RESET)\n' 'Workflow'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'cluster-viz' 'Run Java + Python pipelines, write outputs/, open PNGs in a viewer'
	@printf '\n'
	@printf '$(BOLD)$(MAG)%s$(RESET)\n' 'Steps (same as cluster-viz, without opening files)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'outputs-dir' 'Ensure outputs/ exists'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'java-viz' 'Tribuo only → $(DIM)outputs/java_tribuo_hdbscan.png|json$(RESET)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'python-viz' 'UMAP + HDBSCAN only → $(DIM)outputs/python_umap_hdbscan.png|json$(RESET)'
	@printf '\n'
	@printf '$(BOLD)$(MAG)%s$(RESET)\n' 'Cleanup'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'clean' 'mvn clean, Python $(DIM)__pycache__$(RESET) / egg-info / build stamps $(DIM)(keeps .venv)$(RESET)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'clean-data' 'Remove $(DIM)outputs/$(RESET) (plots + JSON)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'clean-venv' 'Remove $(DIM)python/.venv/$(RESET)'
	@printf '\n'
	@printf '$(BOLD)$(YELLOW)%s$(RESET)\n' 'Variables'
	@printf '  $(WHITE)PROMPTS_FILE$(RESET)  default $(DIM)$(PROMPTS_FILE)$(RESET)\n'
	@printf '  $(WHITE)INTERACTIVE$(RESET)   $(DIM)1$(RESET) = also show matplotlib window on Python $(DIM)(default from SHOW, else 0)$(RESET)\n'
	@printf '  $(WHITE)SHOW$(RESET)            $(DIM)1$(RESET) = same as $(DIM)INTERACTIVE=1$(RESET) when INTERACTIVE is unset\n'
	@printf '\n'
	@printf '$(DIM)%s$(RESET)\n' 'Examples: make cluster-viz   ·   PROMPTS_FILE=./data/prompts.txt make cluster-viz   ·   INTERACTIVE=1 make python-viz'

# Create venv when missing; install editable package only when pyproject.toml is newer than the stamp.
$(VENV_PYTHON):
	@cd "$(PYTHON_DIR)" && python3 -m venv "$(VENV_DIR)"

$(VENV_STAMP): $(PYTHON_DIR)/pyproject.toml | $(VENV_PYTHON)
	@cd "$(PYTHON_DIR)" && . "$(VENV_DIR)/bin/activate" && pip install -q -e .
	@touch "$(VENV_STAMP)"

# Build artifacts + Python packaging caches (does not remove python/.venv).
clean:
	@cd "$(REPO_ROOT)/java" && mvn -q clean
	@find "$(PYTHON_DIR)" -path "$(PYTHON_DIR)/.venv" -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf "$(PYTHON_DIR)"/*.egg-info "$(PYTHON_DIR)/build" "$(PYTHON_DIR)/dist"
	@rm -f "$(VENV_STAMP)"

# Generated plots and sidecar JSON under outputs/ (does not touch data/prompts.txt).
clean-data:
	@rm -rf "$(OUTPUT_DIR)"

# Remove the Python virtualenv (next python-viz / cluster-viz recreates it).
clean-venv:
	@rm -rf "$(VENV_DIR)"

cluster-viz: outputs-dir java-viz python-viz
	@echo "Opening outputs (if a viewer is available)..."
	@$(OPEN_CMD) "$(OUTPUT_DIR)/java_tribuo_hdbscan.png" 2>/dev/null || true
	@$(OPEN_CMD) "$(OUTPUT_DIR)/python_umap_hdbscan.png" 2>/dev/null || true

outputs-dir:
	@mkdir -p "$(OUTPUT_DIR)"

java-viz:
	@cd "$(REPO_ROOT)/java" && mvn -q compile exec:java \
		-Dexec.args="$(PROMPTS_FILE) $(OUTPUT_DIR)/java_tribuo_hdbscan.png"

python-viz: $(VENV_STAMP)
	@cd "$(PYTHON_DIR)" && \
		. "$(VENV_DIR)/bin/activate" && \
		if [ "$(INTERACTIVE)" = "1" ]; then \
			python prompt_cluster.py --prompts "$(PROMPTS_FILE)" --output "$(OUTPUT_DIR)/python_umap_hdbscan.png" --show; \
		else \
			python prompt_cluster.py --prompts "$(PROMPTS_FILE)" --output "$(OUTPUT_DIR)/python_umap_hdbscan.png"; \
		fi
