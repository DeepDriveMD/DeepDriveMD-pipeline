SHELL=/bin/bash

out_dir="/p/gpfs1/${USER}/Outputs"

pdir:
	mkdir -p ${out_dir}/${d}/postproduction
rmsd: pdir
	[[ ! -z "$d" ]] && echo "d = $d" && python rmsd.py ${out_dir}/${d}/aggregation_runs/stage0000 ${out_dir}/${d}/postproduction/
positions: pdir
	[[ ! -z "$d" ]] && echo "d = $d" && python positions.py ${out_dir}/${d}/aggregation_runs/stage0000 ${out_dir}/${d}/postproduction/ $t
loss:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && cd `ls -tr | tail -1` && cd pilot* && python ${PWD}/loss.py task.${d}/*.out
loss1:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && cd ${d1} && cd pilot* && python ${PWD}/loss.py task.${d}/*.out
loss_real:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && cd `ls -tr | tail -1` && cd pilot* && python ${PWD}/loss_real.py task.${d}/*.out
timers:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && python ${PWD}/timers.py ./`ls -tr | tail -1`/pilot.0000
timers1:
	cd /p/gpfs1/${USER}/radical.pilot.sandbox && python ${PWD}/timers.py ${d}/pilot.0000
project: pdir
	PYTHONPATH=/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/deepdrivemd/misc:${PYTHONPATH} python ../deepdrivemd/agents/stream/dbscan.py -c ${out_dir}/${d}/agent_runs/stage0000/task0000/stage0000_task0000.yaml -p
project_mini: pdir
	PYTHONPATH=/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/deepdrivemd/misc:${PYTHONPATH} python ../deepdrivemd/agents/stream/dbscan.py -c ${out_dir}/${d}/agent_runs/stage0000/task0000/stage0000_task0000.yaml -m
tsne_3D: pdir
	PYTHONPATH=/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/deepdrivemd/misc:${PYTHONPATH} python ../deepdrivemd/agents/stream/dbscan.py -c ${out_dir}/${d}/agent_runs/stage0000/task0000/stage0000_task0000.yaml -t
tsne_2D: pdir
	PYTHONPATH=/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/deepdrivemd/misc:${PYTHONPATH} python ../deepdrivemd/agents/stream/dbscan.py -c ${out_dir}/${d}/agent_runs/stage0000/task0000/stage0000_task0000.yaml -T

clean:
	rm -rf *~ *.out *.err *.csv *.npy re.session*
