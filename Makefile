run:
	python  -m deepdrivemd.deepdrivemd_stream -c test/bba/lassen-keras-dbscan-stream.yaml
push:
	bin/push.sh ${m}
clean:
	rm -rf /usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1
	rm -f *~ */*~
	rm -rf __pycache__ */__pycache__
