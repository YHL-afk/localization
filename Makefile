# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++11 -c
LDFLAGS = -Wall
LIBS = -lboost_system -lpthread -lpython3.9

# Source files and objects
SRCS = lds_driver.cpp
OBJS = $(SRCS:.cpp=.o)

# Include directories
INC_DIRS = -I/usr/include \
           -I/usr/include/python3.9 \
           -I$(HOME)/.local/lib/python3.9/site-packages/numpy/core/include \
           -I./include

# Library directories
LIB_DIRS = -L/usr/lib/x86_64-linux-gnu

# Output target
TARGET = lds_driver

# Default build target
all: $(TARGET)

# Link the final target
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LIB_DIRS) $(LIBS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC_DIRS) $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)