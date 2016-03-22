TARGET = reaction
LIBS = #-larmadillo -lconfig++ 
CC = nvcc
CFLAGS = -O3 #-pg -g -larmadillo -fopenmp
ODIR = obj
SRCDIR = src

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(ODIR)/%.o, $(wildcard $(SRCDIR)/*.cpp))
OBJECTS_CU = $(patsubst $(SRCDIR)/%.cu, $(ODIR)/%.o, $(wildcard $(SRCDIR)/*.cu))
HEADERS = $(wildcard *.h)

$(ODIR)/%.o: $(SRCDIR)/%.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

$(ODIR)/%.o: $(SRCDIR)/%.cu 
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS) $(OBJECTS_CU)

$(TARGET): $(OBJECTS) $(OBJECTS_CU)
	$(CC) -pg -g $(OBJECTS) $(OBJECTS_CU) $(LIBS) -o $@

clean:
	-rm -f $(ODIR)/*.o
	-rm -f $(TARGET)
