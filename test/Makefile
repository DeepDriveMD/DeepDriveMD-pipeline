SHELL=/bin/bash

run1:
	../bin/run.sh test1_stream
run2:
	../bin/run.sh lassen-keras-dbscan_stream
run3:	
	../bin/run.sh lassen-keras-dbscan_stream_noutliers
run3m:	
	../bin/run.sh lassen-keras-dbscan_stream_noutliers_mini
run4:	
	../bin/run.sh lassen-keras-dbscan_stream_random_outliers
run4m:
	../bin/run.sh lassen-keras-dbscan_stream_random_outliers_mini
run5:
	../bin/run.sh lassen-keras-dbscan_stream_greedy
run5m:
	../bin/run.sh lassen-keras-dbscan_stream_greedy_mini
run6:
	../bin/run.sh lassen-keras-dbscan_stream_smoothended_rec
run6a:
	../bin/run.sh lassen-aae_stream_smoothended_rec
run7:
	../bin/run.sh lassen-keras-dbscan_stream_insRec_OM_region
run7m:
	../bin/run.sh lassen-keras-dbscan_stream_insRec_OM_region_mini
run8:
	../bin/run.sh lassen-keras-dbscan_stream_spike
run9:
	../bin/run.sh lassen-keras-dbscan_stream_smoothended_rec_120
run9a:
	../bin/run.sh lassen-aae_stream_smoothended_rec_120
run10:
	../bin/run.sh lassen-keras-dbscan_stream_multi-ligand
run11:
	../bin/run.sh lassen-keras-dbscan_stream_multi-ligand_120
clean:
	rm -f *~ */*~
	rm -rf __pycache__ */__pycache__ *.log */*/__pycache__ */*/*/__pycache__
	rm -rf re.session.*
	[[ ! -z "$d" ]] && echo "d = $d" && rm -rf /p/gpfs1/yakushin/Outputs/${d}


