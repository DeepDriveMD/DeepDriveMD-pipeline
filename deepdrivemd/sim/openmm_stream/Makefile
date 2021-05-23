run:
	PYTHONPATH=/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/deepdrivemd/misc:${PYTHONPATH} python run_openmm.py -c md.yaml
clean:
	rm -rf *~ __pycache__ /usr/workspace/cv_ddmd/yakushin/Integration1/Outlier_Search/*.lock /usr/workspace/cv_ddmd/yakushin/Integration1/Outlier_Search/*lassen* 
	rm -rf /usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1/molecular_dynamics_runs/stage0000/task0000/md_output.bp 
	rm -rf /usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1/molecular_dynamics_runs/stage0000/task0000/{0,1,2,3,4,5,6,7,8,9}*
	rm -rf /usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1/molecular_dynamics_runs/stage0000/task0000/*.pdb
