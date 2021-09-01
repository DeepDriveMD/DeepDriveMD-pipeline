#SHELL=/bin/bash
.DEFAULT_GOAL := all
isort = isort deepdrivemd test
black = black --target-version py37 deepdrivemd test

.PHONY: venv
venv:
	python -m venv .venv

.PHONY: install-dev
install-dev:
	python -m pip install --upgrade wheel pip
	python -m pip install -r requirements_dev.txt

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	flake8 deepdrivemd/ test/
	$(isort) --check-only --df
	$(black) --check --diff

.PHONY: mypy
mypy:
	mypy --config-file setup.cfg --package deepdrivemd
	# mypy --config-file setup.cfg test/

.PHONY: all
all: format lint mypy

# Streaming tests
run1:
	bin/run.sh test1_stream
run2:
	bin/run.sh lassen-keras-dbscan_stream
run3:	
	bin/run.sh lassen-keras-dbscan_stream_noutliers
run4:	
	bin/run.sh lassen-keras-dbscan_stream_random_outliers
run5:
	bin/run.sh lassen-keras-dbscan_stream_greedy
run6:
	bin/run.sh lassen-keras-dbscan_stream_smoothended_rec
run7:
	bin/run.sh lassen-keras-dbscan_stream_insRec_OM_region
watch1:
	watch bpls ../Outputs/${d}/aggregation_runs/stage0000/task0000/agg.bp
watch2:
	watch ls -ltr ../Outputs/${d}/molecular_dynamics_runs/stage0000/task0000/
watch3:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && cd `ls -tr | tail -1` && cd pilot* && tail -f task.${d}/*.out
clean:
	rm -f *~ */*~
	rm -rf __pycache__ */__pycache__ *.log */*/__pycache__ */*/*/__pycache__
	rm -rf re.session.*
	[[ ! -z "$d" ]] && echo "d = $d" && rm -rf ../Outputs/${d}