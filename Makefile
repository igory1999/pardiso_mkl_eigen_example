run:	t
	mpirun -n 2 ./t
t:	t.o
	mpiicpc -o t -std=c++11 -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -liomp5 -lpthread  -ldl t.o -lm
t.o:	t.C
	mpiicpc -c -std=c++11 -DMKL_ILP64 -I${MKLROOT}/include -I${EIGEN3_INCLUDE_DIR}/eigen3 t.C
clean:
	rm -f *.o t *~

