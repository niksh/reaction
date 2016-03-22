CUDA_INSTALL_PATH	:= /usr/local/cuda
CUDA_ARCH 			:= sm_30

INCLUDES			+= -I$(CUDA_INSTALL_PATH)/include
LIBS				+= -L$(CUDA_INSTALL_PATH)/lib64
CFLAGS				:= -O3 -g $(INCLUDES)
LDFLAGS				:= $(LIBS) -lrt -lm -lcudart

# compilers
NVCC				:= $(CUDA_INSTALL_PATH)/bin/nvcc -arch $(CUDA_ARCH) --ptxas-options=-v -use_fast_math
CXX					:= g++
LINKER				:= g++ -fPIC

OBJDIR				:= obj
SRCDIR				:= src
include common.mk

# files
REACTION_SOURCES	:= \
	$(SRCDIR)/dcdio.cpp \
    $(SRCDIR)/main.cu \
    $(SRCDIR)/timer.cpp \
    $(SRCDIR)/xyzio.cpp \
    $(SRCDIR)/wrapper.cpp 

all: reaction

REACTION_OBJS			:= $(call objects, $(REACTION_SOURCES))
-include $(REACTION_OBJS:.o=.d)
 
reaction: $(REACTION_OBJS)
	$(LINKER) -o $@ $(REACTION_OBJS) $(LDFLAGS)

clean:
	rm -f reaction
	rm -rf "$(OBJDIR)"
	
.PHONY: makedirs clean all
