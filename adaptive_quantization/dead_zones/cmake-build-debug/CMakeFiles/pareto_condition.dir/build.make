# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/gengxue/Downloads/clion-2018.1.5/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/gengxue/Downloads/clion-2018.1.5/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gengxue/git/adder_trees

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gengxue/git/adder_trees/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/pareto_condition.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pareto_condition.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pareto_condition.dir/flags.make

CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o: CMakeFiles/pareto_condition.dir/flags.make
CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o: ../pareto_condition.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gengxue/git/adder_trees/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o -c /home/gengxue/git/adder_trees/pareto_condition.cpp

CMakeFiles/pareto_condition.dir/pareto_condition.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pareto_condition.dir/pareto_condition.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gengxue/git/adder_trees/pareto_condition.cpp > CMakeFiles/pareto_condition.dir/pareto_condition.cpp.i

CMakeFiles/pareto_condition.dir/pareto_condition.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pareto_condition.dir/pareto_condition.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gengxue/git/adder_trees/pareto_condition.cpp -o CMakeFiles/pareto_condition.dir/pareto_condition.cpp.s

CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o.requires:

.PHONY : CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o.requires

CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o.provides: CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o.requires
	$(MAKE) -f CMakeFiles/pareto_condition.dir/build.make CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o.provides.build
.PHONY : CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o.provides

CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o.provides.build: CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o


# Object files for target pareto_condition
pareto_condition_OBJECTS = \
"CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o"

# External object files for target pareto_condition
pareto_condition_EXTERNAL_OBJECTS =

pareto_condition: CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o
pareto_condition: CMakeFiles/pareto_condition.dir/build.make
pareto_condition: CMakeFiles/pareto_condition.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gengxue/git/adder_trees/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pareto_condition"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pareto_condition.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pareto_condition.dir/build: pareto_condition

.PHONY : CMakeFiles/pareto_condition.dir/build

CMakeFiles/pareto_condition.dir/requires: CMakeFiles/pareto_condition.dir/pareto_condition.cpp.o.requires

.PHONY : CMakeFiles/pareto_condition.dir/requires

CMakeFiles/pareto_condition.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pareto_condition.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pareto_condition.dir/clean

CMakeFiles/pareto_condition.dir/depend:
	cd /home/gengxue/git/adder_trees/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gengxue/git/adder_trees /home/gengxue/git/adder_trees /home/gengxue/git/adder_trees/cmake-build-debug /home/gengxue/git/adder_trees/cmake-build-debug /home/gengxue/git/adder_trees/cmake-build-debug/CMakeFiles/pareto_condition.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pareto_condition.dir/depend

