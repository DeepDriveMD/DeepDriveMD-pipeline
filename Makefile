SHELL=/bin/bash

run1:
	../bin/run.sh test1_stream
run2:
	../bin/run.sh lassen-keras-dbscan_stream
push:
	bin/push.sh ${m}
watch1:
	watch bpls ../Outputs/${d}/aggregation_runs/stage0000/task0000/agg.bp
watch2:
	watch ls -ltr ../Outputs/${d}/molecular_dynamics_runs/stage0000/task0000/
watch3:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && cd `ls -tr | tail -1` && cd pilot* && tail -f task.${d}/*.out
clean:
	rm -f *~ */*~
	rm -rf __pycache__ */__pycache__ *.log
	rm -rf re.session.*
	[[ ! -z "$d" ]] && echo "d = $d" && rm -rf ../Outputs/${d}



