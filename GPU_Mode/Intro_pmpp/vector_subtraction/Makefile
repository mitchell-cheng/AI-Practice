NVCC        = /usr/local/cuda/bin/nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = vector_subtraction
OBJ	        = vector_subtraction.o

default: $(EXE)

vector_subtraction.o: vector_subtraction.cu
		$(NVCC) -c -o $@ vector_subtraction.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
		$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
		rm -Force *.o $(EXE)