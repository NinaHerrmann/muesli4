for size in 100 200 300 400 500 600 700 800 900; do
	bin/dc_test_seq $size 0 
	for np in 1 2; do
		for split in 0.1 0.2 0.3 0.4 0.5; do
			mpirun -np $np bin/dc_test_gpu $size 1 $split 
			mpirun -np $np bin/dc_test_gpu $size 2 $split
		done
		mpirun -np $np bin/dc_test_cpu $size 0 1 
	done
done
