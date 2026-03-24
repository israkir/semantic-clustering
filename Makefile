# Semantic-clustering: same stack as the tool-mismatch clustering service defaults
# (BGE-small ONNX, Tribuo HDBSCAN*, Smile UMAP plot, text normalize per application.yaml).

REPO_ROOT := $(abspath .)
OUTPUT_DIR := $(REPO_ROOT)/outputs
PYTHON_DIR := $(REPO_ROOT)/python
VENV_DIR := $(PYTHON_DIR)/.venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_STAMP := $(VENV_DIR)/.install-stamp
PROMPTS_FILE ?= $(REPO_ROOT)/data/prompts.txt
JAVA_VIZ_PNG := $(OUTPUT_DIR)/java_tribuo_hdbscan.png
PYTHON_VIZ_PNG := $(OUTPUT_DIR)/python_umap_hdbscan.png
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
CYAN   := $(_esc)[36m
WHITE  := $(_esc)[37m
RESET  := $(_esc)[0m

.DEFAULT_GOAL := help

ONNX_DIR := $(REPO_ROOT)/model/onnx/bge-small-en-v1.5
ONNX_MODEL := $(ONNX_DIR)/model.onnx
ONNX_VOCAB := $(ONNX_DIR)/vocab.txt

.PHONY: help cluster-viz outputs-dir git-lfs-pull ensure-onnx-model java-viz python-viz clean clean-data clean-venv

help:
	@printf '$(BOLD)$(CYAN)%s$(RESET)\n' 'semantic-clustering'
	@printf '$(DIM)%s$(RESET)\n' '  Java: BGE-small ONNX → Tribuo HDBSCAN* → Smile UMAP (tool-mismatch defaults).'
	@printf '$(DIM)%s$(RESET)\n' '  Python: MiniLM + UMAP + HDBSCAN — comparison baseline only.'
	@printf '\n'
	@printf '$(BOLD)%s$(RESET)\n' 'Targets'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'java-viz' 'Java only → $(DIM)outputs/java_tribuo_hdbscan.png|json$(RESET)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'python-viz' 'Python only → $(DIM)outputs/python_umap_hdbscan.png|json$(RESET)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'cluster-viz' 'Run $(DIM)java-viz$(RESET) + $(DIM)python-viz$(RESET); open both PNGs'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'outputs-dir' 'Create $(DIM)outputs/$(RESET)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'git-lfs-pull' '$(DIM)git lfs pull$(RESET) (ONNX weights)'
	@printf '\n'
	@printf '$(BOLD)%s$(RESET)\n' 'Cleanup'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'clean' '$(DIM)mvn clean$(RESET); Python caches $(DIM)(keeps .venv)$(RESET)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'clean-data' 'Remove $(DIM)outputs/$(RESET)'
	@printf '  $(GREEN)%-18s$(RESET) %s\n' 'clean-venv' 'Remove $(DIM)python/.venv/$(RESET)'
	@printf '\n'
	@printf '$(WHITE)PROMPTS_FILE$(RESET)=$(DIM)$(PROMPTS_FILE)$(RESET)  $(WHITE)INTERACTIVE$(RESET)=$(DIM)1$(RESET) for Python $(DIM)matplotlib --show$(RESET)\n'

$(VENV_PYTHON):
	@cd "$(PYTHON_DIR)" && python3 -m venv "$(VENV_DIR)"

$(VENV_STAMP): $(PYTHON_DIR)/pyproject.toml | $(VENV_PYTHON)
	@cd "$(PYTHON_DIR)" && . "$(VENV_DIR)/bin/activate" && pip install -q -e .
	@touch "$(VENV_STAMP)"

outputs-dir:
	@mkdir -p "$(OUTPUT_DIR)"

git-lfs-pull:
	@if command -v git-lfs >/dev/null 2>&1; then \
		printf '%s\n' "$(DIM)Pulling Git LFS files (ONNX, etc.)…$(RESET)"; \
		(cd "$(REPO_ROOT)" && git lfs install && git lfs pull) \
			|| printf '%s\n' "$(YELLOW)warning:$(RESET) git lfs pull failed."; \
	else \
		printf '%s\n' "$(YELLOW)warning:$(RESET) git-lfs not installed. See https://git-lfs.com"; \
	fi

ensure-onnx-model: git-lfs-pull
	@if [ ! -f "$(ONNX_MODEL)" ] || [ ! -f "$(ONNX_VOCAB)" ]; then \
		printf '%s\n' "$(RED)error:$(RESET) need $(ONNX_MODEL) and $(ONNX_VOCAB)"; \
		printf '%s\n' "  Run $(BOLD)make git-lfs-pull$(RESET) after clone."; \
		exit 1; \
	fi

java-viz: outputs-dir ensure-onnx-model
	@cd "$(REPO_ROOT)/java" && mvn -q compile exec:java \
		-Dexec.mainClass=dev.semanticclustering.PromptClusterPipeline \
		-Dexec.args="$(PROMPTS_FILE) $(JAVA_VIZ_PNG)"

python-viz: outputs-dir $(VENV_STAMP)
	@cd "$(PYTHON_DIR)" && \
		. "$(VENV_DIR)/bin/activate" && \
		if [ "$(INTERACTIVE)" = "1" ]; then \
			python prompt_cluster.py --prompts "$(PROMPTS_FILE)" --output "$(PYTHON_VIZ_PNG)" --show; \
		else \
			python prompt_cluster.py --prompts "$(PROMPTS_FILE)" --output "$(PYTHON_VIZ_PNG)"; \
		fi

cluster-viz: outputs-dir java-viz python-viz
	@echo "Opening outputs (if a viewer is available)..."
	@$(OPEN_CMD) "$(JAVA_VIZ_PNG)" 2>/dev/null || true
	@$(OPEN_CMD) "$(PYTHON_VIZ_PNG)" 2>/dev/null || true

clean:
	@cd "$(REPO_ROOT)/java" && mvn -q clean
	@find "$(PYTHON_DIR)" -path "$(PYTHON_DIR)/.venv" -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf "$(PYTHON_DIR)"/*.egg-info "$(PYTHON_DIR)/build" "$(PYTHON_DIR)/dist"
	@rm -f "$(VENV_STAMP)"

clean-data:
	@rm -rf "$(OUTPUT_DIR)"

clean-venv:
	@rm -rf "$(VENV_DIR)"
