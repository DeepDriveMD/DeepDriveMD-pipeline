run:
	./run.sh
push:
	bin/push.sh ${m}
clean:
	rm -rf /usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1
	rm -f *~ */*~
	rm -rf __pycache__ */__pycache__ *.log
	rm -rf re.session.*
	rm -rf ../Outlier_Search/lassen*
	rm -rf ../Outlier_Search/*.lock 
edit1:
	emacs -nw deepdrivemd/deepdrivemd_stream.py
edit2:
	emacs -nw test/bba/lassen-keras-dbscan-stream.yaml
out:
	ls -ltr /usr/workspace/cv_ddmd/yakushin/Integration1/Outputs/1
