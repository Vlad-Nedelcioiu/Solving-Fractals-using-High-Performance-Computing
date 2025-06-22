CC=gcc
MPICC=mpicc
CFLAGS=-O2 -Wall
OMPFLAGS=-fopenmp

all: julia_openmp julia_mpi

julia_openmp: julia_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o julia_openmp julia_openmp.c

julia_mpi: julia_mpi.c
	$(MPICC) $(CFLAGS) -o julia_mpi julia_mpi.c
	mpirun -np 4 ./julia_mpi

julia_hybrid: julia_hybrid.c
	$(MPICC) $(CFLAGS) -fopenmp -o julia_hybrid julia_hybrid.c
	mpirun -np 6 ./julia_hybrid

clean:
	rm -f julia_openmp julia_mpi julia_hybrid
	rm -f *.txt
	rm -f logs/*.txt
	rm -f summaries/*.txt
	rm -f images/*.ppm
	rm -f *.png
	rm -f gmon.out massif.out.* cachegrind.out.*

plot:
	python3 plot_chunks.py

massif:
	valgrind --tool=massif ./julia
	ms_print massif.out.* > heap_report.txt

cachegrind:
	valgrind --tool=cachegrind ./julia
	cg_annotate cachegrind.out.* > cache_report.txt

#pg:
#	gcc -pg -g -fopenmp -o julia julia_openmp.c
#	./julia
#	gprof julia gmon.out > profile_gprof.txt
	

resources:
	/usr/bin/time -v ./julia 2> profile_time.txt
