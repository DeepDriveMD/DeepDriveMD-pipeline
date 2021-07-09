run:
	bin/run.sh
push:
	bin/push.sh ${m}
clean:
	rm -f *~ */*~
	rm -rf __pycache__ */__pycache__ *.log
	rm -rf re.session.*
	rm -rf ../Outputs/${d}



