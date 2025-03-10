# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jakey/geo/containerized_host

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jakey/geo/containerized_host/build

# Include any dependencies generated for this target.
include CMakeFiles/cuda_ptx_runner.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cuda_ptx_runner.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_ptx_runner.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_ptx_runner.dir/flags.make

CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o: CMakeFiles/cuda_ptx_runner.dir/flags.make
CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o: ../tiny_ptx.c
CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o: CMakeFiles/cuda_ptx_runner.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jakey/geo/containerized_host/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o -MF CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o.d -o CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o -c /home/jakey/geo/containerized_host/tiny_ptx.c

CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jakey/geo/containerized_host/tiny_ptx.c > CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.i

CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jakey/geo/containerized_host/tiny_ptx.c -o CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.s

# Object files for target cuda_ptx_runner
cuda_ptx_runner_OBJECTS = \
"CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o"

# External object files for target cuda_ptx_runner
cuda_ptx_runner_EXTERNAL_OBJECTS =

libcuda_ptx_runner.so: CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o
libcuda_ptx_runner.so: CMakeFiles/cuda_ptx_runner.dir/build.make
libcuda_ptx_runner.so: /usr/lib/x86_64-linux-gnu/libcudart_static.a
libcuda_ptx_runner.so: /usr/lib/x86_64-linux-gnu/librt.a
libcuda_ptx_runner.so: /usr/lib/x86_64-linux-gnu/libcuda.so
libcuda_ptx_runner.so: CMakeFiles/cuda_ptx_runner.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jakey/geo/containerized_host/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library libcuda_ptx_runner.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_ptx_runner.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_ptx_runner.dir/build: libcuda_ptx_runner.so
.PHONY : CMakeFiles/cuda_ptx_runner.dir/build

CMakeFiles/cuda_ptx_runner.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_ptx_runner.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_ptx_runner.dir/clean

CMakeFiles/cuda_ptx_runner.dir/depend:
	cd /home/jakey/geo/containerized_host/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jakey/geo/containerized_host /home/jakey/geo/containerized_host /home/jakey/geo/containerized_host/build /home/jakey/geo/containerized_host/build /home/jakey/geo/containerized_host/build/CMakeFiles/cuda_ptx_runner.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda_ptx_runner.dir/depend

