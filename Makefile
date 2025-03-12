# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++11 -c
LDFLAGS = -Wall

# 自动获取 Python 配置信息
PYTHON_LIBS = $(shell python3-config --ldflags)
PYTHON_INCLUDES = $(shell python3-config --includes)
NUMPY_INCLUDES = $(shell python3 -c "import numpy; print(numpy.get_include())")

# Libraries
LIBS = -lboost_system -lpthread $(PYTHON_LIBS)

# Source files and objects
SRCS = lds_driver.cpp
OBJS = $(SRCS:.cpp=.o)

# Include directories
INC_DIRS = $(PYTHON_INCLUDES) -I$(NUMPY_INCLUDES) -I./include

# Library directories 
LIB_DIRS =

# Output target
TARGET = lds_driver

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LIB_DIRS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC_DIRS) $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
