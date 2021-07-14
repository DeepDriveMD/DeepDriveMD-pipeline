SHELL=/bin/bash

rmsd:
	[[ ! -z "$d" ]] && echo "d = $d" && python rmsd.py ../../Outputs/${d}/aggregation_runs/stage0000 /tmp
loss:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && cd `ls -tr | tail -1` && cd pilot* && python ${PWD}/loss.py task.${d}/*.out
loss_real:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && cd `ls -tr | tail -1` && cd pilot* && python ${PWD}/loss_real.py task.${d}/*.out
timers:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && python ${PWD}/timers.py ./`ls -tr | tail -1`/pilot*

clean:
	rm -f *~ *.out *.err *.csv *.npy
