# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/jakey/geo/containerized_host/build/CMakeFiles /home/jakey/geo/containerized_host/build//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/jakey/geo/containerized_host/build/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named cuda_ptx_runner

# Build rule for target.
cuda_ptx_runner: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 cuda_ptx_runner
.PHONY : cuda_ptx_runner

# fast build rule for target.
cuda_ptx_runner/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda_ptx_runner.dir/build.make CMakeFiles/cuda_ptx_runner.dir/build
.PHONY : cuda_ptx_runner/fast

#=============================================================================
# Target rules for targets named tiny_ptx

# Build rule for target.
tiny_ptx: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 tiny_ptx
.PHONY : tiny_ptx

# fast build rule for target.
tiny_ptx/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tiny_ptx.dir/build.make CMakeFiles/tiny_ptx.dir/build
.PHONY : tiny_ptx/fast

tiny_ptx.o: tiny_ptx.c.o
.PHONY : tiny_ptx.o

# target to build an object file
tiny_ptx.c.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda_ptx_runner.dir/build.make CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tiny_ptx.dir/build.make CMakeFiles/tiny_ptx.dir/tiny_ptx.c.o
.PHONY : tiny_ptx.c.o

tiny_ptx.i: tiny_ptx.c.i
.PHONY : tiny_ptx.i

# target to preprocess a source file
tiny_ptx.c.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda_ptx_runner.dir/build.make CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tiny_ptx.dir/build.make CMakeFiles/tiny_ptx.dir/tiny_ptx.c.i
.PHONY : tiny_ptx.c.i

tiny_ptx.s: tiny_ptx.c.s
.PHONY : tiny_ptx.s

# target to generate assembly for a file
tiny_ptx.c.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cuda_ptx_runner.dir/build.make CMakeFiles/cuda_ptx_runner.dir/tiny_ptx.c.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/tiny_ptx.dir/build.make CMakeFiles/tiny_ptx.dir/tiny_ptx.c.s
.PHONY : tiny_ptx.c.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... cuda_ptx_runner"
	@echo "... tiny_ptx"
	@echo "... tiny_ptx.o"
	@echo "... tiny_ptx.i"
	@echo "... tiny_ptx.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

