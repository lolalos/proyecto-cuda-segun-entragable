INCDIR = -I.
DBG    = -g
OPT    = -O3
CUDA   = nvcc
CFLAGS = $(DBG) $(OPT) $(INCDIR)
LINK   = -lm 

.cu.o:
	$(CUDA) $(CFLAGS) -c $< -o $@

all: segment

segment: segment.cu segment-image.cuh segment-graph.h disjoint-set.h
	$(CUDA) $(CFLAGS) -o segment segment.cu $(LINK)

clean:
	/bin/rm -f segment *.o

clean-all: clean
	/bin/rm -f *~
