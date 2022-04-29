for size in 10 20 30 40 50; do
	bin/dc_test_seq $size 0 1 0 1000 
	for np in 1 2; do
		for split in 0.1 0.2 0.3 0.4 0.5; do
			mpirun -np $np bin/dc_test_gpu $size 1 $split 0 1000
			mpirun -np $np bin/dc_test_gpu $size 2 $split 0 1000
		done
		mpirun -np $np bin/dc_test_cpu $size 0 1 0 1000
	done
done
