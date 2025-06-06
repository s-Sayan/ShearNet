# Makefile for ShearNet installation

# Ensure we use bash for consistent echo behavior
SHELL := /bin/bash

# Color definitions
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m # No Color

# Bold variants
BOLD := \033[1m
BOLD_RED := \033[1;31m
BOLD_GREEN := \033[1;32m

# Conda source command
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh

.PHONY: help install install-gpu install-gpu12 install-dev install-all clean test uninstall

help:
	@printf "\n"
	@printf "$(BOLD)$(CYAN)╔════════════════════════════════════════════╗$(NC)\n"
	@printf "$(BOLD)$(CYAN)║       ShearNet Installation Options        ║$(NC)\n"
	@printf "$(BOLD)$(CYAN)╚════════════════════════════════════════════╝$(NC)\n"
	@printf "\n"
	@printf "$(BOLD)$(YELLOW)Basic Installation:$(NC)\n"
	@printf "  $(CYAN)make install$(NC)       - Install CPU version\n"
	@printf "  $(CYAN)make install-gpu$(NC)   - Install GPU version (CUDA 12)\n"
	@printf "\n"
	@printf "$(BOLD)$(YELLOW)Development Installation:$(NC)\n"
	@printf "  $(CYAN)make install-dev$(NC)   - Install with development tools\n"
	@printf "  $(CYAN)make install-all$(NC)   - Install GPU + dev tools\n"
	@printf "\n"
	@printf "$(BOLD)$(YELLOW)Maintenance:$(NC)\n"
	@printf "  $(CYAN)make test$(NC)          - Run tests\n"
	@printf "  $(CYAN)make clean$(NC)         - Remove ALL ShearNet environments\n"
	@printf "  $(CYAN)make uninstall$(NC)     - Remove specific environment\n"
	@printf "\n"
	@printf "$(BOLD)$(GREEN)Example:$(NC) make install-gpu\n"
	@printf "\n"

install:
	@printf "\n"
	@printf "$(BOLD)$(BLUE)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(WHITE)     Installing ShearNet (CPU Version)      $(NC)\n"
	@printf "$(BOLD)$(BLUE)════════════════════════════════════════════$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Creating conda environment 'shearnet'...$(NC)\n"
	@conda create -n shearnet python=3.11 pip numba -y --no-default-packages -q
	@printf "$(GREEN)✓ Environment created$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Installing ShearNet package and dependencies...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet && \
		pip install -e . --quiet && \
		printf "$(GREEN)✓ ShearNet installed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Installing NGmix...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet && \
		pip install git+https://github.com/esheldon/ngmix.git --use-pep517 --quiet && \
		printf "$(GREEN)✓ NGmix installed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Running post-installation setup...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet && \
		python scripts/post_installation.py
	@printf "$(GREEN)✓ Setup complete$(NC)\n"
	@printf "\n"
	@printf "$(BOLD)$(GREEN)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(GREEN)✓ Installation successful!$(NC)\n"
	@printf "$(BOLD)$(WHITE)  Activate with: $(CYAN)conda activate shearnet$(NC)\n"
	@printf "$(BOLD)$(GREEN)════════════════════════════════════════════$(NC)\n"
	@printf "\n"

install-gpu:
	@printf "\n"
	@printf "$(BOLD)$(PURPLE)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(WHITE)   Installing ShearNet (GPU CUDA 12)        $(NC)\n"
	@printf "$(BOLD)$(PURPLE)════════════════════════════════════════════$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Creating conda environment 'shearnet_gpu'...$(NC)\n"
	@conda create -n shearnet_gpu python=3.11 pip numba -y --no-default-packages -q
	@printf "$(GREEN)✓ Environment created$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Installing ShearNet with GPU support (CUDA 12)...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_gpu && \
		pip install -U -e ".[gpu]" --quiet && \
		printf "$(GREEN)✓ ShearNet GPU version installed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Installing NGmix...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_gpu && \
		pip install git+https://github.com/esheldon/ngmix.git --use-pep517 -q 2>&1 | grep -v DEPRECATION || true && \
		printf "$(GREEN)✓ NGmix installed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Running post-installation setup...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_gpu && \
		python scripts/post_installation.py
	@printf "$(GREEN)✓ Setup complete$(NC)\n"
	@printf "\n"
	@printf "$(BOLD)$(GREEN)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(GREEN)✓ GPU Installation successful!$(NC)\n"
	@printf "$(BOLD)$(WHITE)  Activate with: $(CYAN)conda activate shearnet_gpu$(NC)\n"
	@printf "$(BOLD)$(GREEN)════════════════════════════════════════════$(NC)\n"
	@printf "\n"

install-dev:
	@printf "\n"
	@printf "$(BOLD)$(CYAN)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(WHITE)  Installing ShearNet (Development Mode)    $(NC)\n"
	@printf "$(BOLD)$(CYAN)════════════════════════════════════════════$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Creating conda environment 'shearnet_dev'...$(NC)\n"
	@conda create -n shearnet_dev python=3.11 pip numba -y --no-default-packages -q
	@printf "$(GREEN)✓ Environment created$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Installing ShearNet with development tools...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_dev && \
		pip install -e ".[dev]" --quiet && \
		printf "$(GREEN)✓ ShearNet + dev tools installed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Installing NGmix...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_dev && \
		pip install git+https://github.com/esheldon/ngmix.git --quiet && \
		printf "$(GREEN)✓ NGmix installed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Running post-installation setup...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_dev && \
		python scripts/post_installation.py
	@printf "$(GREEN)✓ Setup complete$(NC)\n"
	@printf "\n"
	@printf "$(BOLD)$(GREEN)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(GREEN)✓ Development installation successful!$(NC)\n"
	@printf "$(BOLD)$(WHITE)  Activate with: $(CYAN)conda activate shearnet_dev$(NC)\n"
	@printf "$(BOLD)$(WHITE)  Tools included: pytest, black, flake8, ipython$(NC)\n"
	@printf "$(BOLD)$(GREEN)════════════════════════════════════════════$(NC)\n"
	@printf "\n"

install-all:
	@printf "\n"
	@printf "$(BOLD)$(PURPLE)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(WHITE)  Installing ShearNet (GPU + Dev Tools)     $(NC)\n"
	@printf "$(BOLD)$(PURPLE)════════════════════════════════════════════$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Creating conda environment 'shearnet_all'...$(NC)\n"
	@conda create -n shearnet_all python=3.11 pip numba cudatoolkit=11.8 -y --no-default-packages -q
	@printf "$(GREEN)✓ Environment created with CUDA support$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Installing ShearNet with all features...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_all && \
		pip install -e ".[gpu,dev]" --quiet && \
		printf "$(GREEN)✓ ShearNet complete installation finished$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Installing NGmix...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_all && \
		pip install git+https://github.com/esheldon/ngmix.git --quiet && \
		printf "$(GREEN)✓ NGmix installed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)→ Running post-installation setup...$(NC)\n"
	@$(CONDA_ACTIVATE) && conda activate shearnet_all && \
		python scripts/post_installation.py
	@printf "$(GREEN)✓ Setup complete$(NC)\n"
	@printf "\n"
	@printf "$(BOLD)$(GREEN)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(GREEN)✓ Complete installation successful!$(NC)\n"
	@printf "$(BOLD)$(WHITE)  Activate with: $(CYAN)conda activate shearnet_all$(NC)\n"
	@printf "$(BOLD)$(WHITE)  Includes: GPU support + all dev tools$(NC)\n"
	@printf "$(BOLD)$(GREEN)════════════════════════════════════════════$(NC)\n"
	@printf "\n"

test:
	@printf "\n"
	@printf "$(BOLD)$(YELLOW)Running ShearNet tests...$(NC)\n"
	@if conda env list | grep -q "shearnet"; then \
		$(CONDA_ACTIVATE) && conda activate shearnet && \
		pytest tests/ -v --color=yes; \
	else \
		printf "$(RED)Error: No ShearNet environment found!$(NC)\n"; \
		printf "$(YELLOW)Run 'make install' first.$(NC)\n"; \
	fi

clean:
	@printf "\n"
	@printf "$(BOLD)$(RED)════════════════════════════════════════════$(NC)\n"
	@printf "$(BOLD)$(RED)     Removing ALL ShearNet Environments     $(NC)\n"
	@printf "$(BOLD)$(RED)════════════════════════════════════════════$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)This will remove:$(NC)\n"
	@printf "  • shearnet\n"
	@printf "  • shearnet_gpu\n"
	@printf "  • shearnet_gpu12\n"
	@printf "  • shearnet_dev\n"
	@printf "  • shearnet_all\n"
	@printf "\n"
	@printf "$(BOLD)$(RED)Are you sure? [y/N] $(NC)"; \
	read -n 1 -r REPLY; \
	printf "\n"; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		printf "$(YELLOW)→ Removing environments...$(NC)\n"; \
		conda env remove -n shearnet -y 2>/dev/null || true; \
		conda env remove -n shearnet_gpu -y 2>/dev/null || true; \
		conda env remove -n shearnet_gpu12 -y 2>/dev/null || true; \
		conda env remove -n shearnet_dev -y 2>/dev/null || true; \
		conda env remove -n shearnet_all -y 2>/dev/null || true; \
		printf "$(GREEN)✓ All environments removed$(NC)\n"; \
	else \
		printf "$(YELLOW)Cancelled$(NC)\n"; \
	fi

uninstall:
	@printf "\n"
	@printf "$(BOLD)$(YELLOW)Available ShearNet environments:$(NC)\n"
	@conda env list | grep shearnet || printf "$(RED)No ShearNet environments found$(NC)\n"
	@printf "\n"
	@printf "$(BOLD)Enter environment name to remove: $(NC)"; \
	read env_name; \
	if conda env list | grep -q "$$env_name"; then \
		printf "$(YELLOW)→ Removing $$env_name...$(NC)\n"; \
		conda env remove -n $$env_name -y; \
		printf "$(GREEN)✓ Environment removed$(NC)\n"; \
	else \
		printf "$(RED)Environment '$$env_name' not found$(NC)\n"; \
	fi