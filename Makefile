PYTHON ?= python3
ARGS ?=
.PHONY: help doctor doctor-single build serve serve-mtp serve-baseline smoke smoke-single bench bench-mtp check dry-run install-hooks
help:
	@printf '%s\n' 'Native llama.cpp Metal router (existing local models; no implicit downloads)' 'make doctor     Start router, validate native INI, list models, stop' 'make build      Build pinned official llama.cpp in .runtime/' 'make serve      Router: local/models.ini, loopback 39483, one resident model' 'make smoke      Router: tiny autoload request, 256K props, unload/child check' 'make serve-mtp  Legacy single Qwen3.8, MTP4, 262144 context' 'make serve-baseline  Legacy single Qwen3.8, MTP off' 'make doctor-single  Legacy Qwen metadata and runner checks' 'make smoke-single   Legacy text and streamed tool/result check' 'make bench      Legacy single-model baseline cold/warm workload' 'make bench-mtp  Legacy paired baseline/MTP lengths 1,2,4' 'make check      Offline standard-library tests' 'make install-hooks  Opt in to repository-local secret/PII checks' 'make dry-run    Print router flags without starting it' 'Override INI with ARGS="--config local/other.ini"; no environment variable needed.' 'Model flags belong in INI, not router CLI. See docs/setup.md.'
doctor doctor-single build serve serve-mtp serve-baseline smoke smoke-single bench bench-mtp:
	$(PYTHON) scripts/recipe.py $@ $(ARGS)
install-hooks:
	sh scripts/install-hooks.sh
check:
	$(PYTHON) -m unittest discover -s tests -v
dry-run:
	$(PYTHON) scripts/recipe.py serve --dry-run $(ARGS)
