run1:
	bin/run.sh test1_stream
run2:
	bin/run.sh lassen-keras-dbscan_stream
push:
	bin/push.sh ${m}
watch1:
	watch bpls ../Outputs/${d}/aggregation_runs/stage0000/task0000/agg.bp
watch2:
	watch ls -ltr ../Outputs/${d}/molecular_dynamics_runs/stage0000/task0000/
clean:
	rm -f *~ */*~
	rm -rf __pycache__ */__pycache__ *.log
	rm -rf re.session.*
	rm -rf ../Outputs/${d}



