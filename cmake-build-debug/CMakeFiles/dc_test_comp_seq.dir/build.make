# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

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
CMAKE_COMMAND = /home/n_herr03@WIWI.UNI-MUENSTER.DE/Programme/clion-2022.2.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/n_herr03@WIWI.UNI-MUENSTER.DE/Programme/clion-2022.2.4/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/dc_test_comp_seq.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/dc_test_comp_seq.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/dc_test_comp_seq.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dc_test_comp_seq.dir/flags.make

CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o: CMakeFiles/dc_test_comp_seq.dir/flags.make
CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o: ../examples/dc_test_comp.cpp
CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o: CMakeFiles/dc_test_comp_seq.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o -MF CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o.d -o CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o -c /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/examples/dc_test_comp.cpp

CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/examples/dc_test_comp.cpp > CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.i

CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/examples/dc_test_comp.cpp -o CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.s

# Object files for target dc_test_comp_seq
dc_test_comp_seq_OBJECTS = \
"CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o"

# External object files for target dc_test_comp_seq
dc_test_comp_seq_EXTERNAL_OBJECTS =

../dc_test_comp/dc_test_comp_seq: CMakeFiles/dc_test_comp_seq.dir/examples/dc_test_comp.cpp.o
../dc_test_comp/dc_test_comp_seq: CMakeFiles/dc_test_comp_seq.dir/build.make
../dc_test_comp/dc_test_comp_seq: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
../dc_test_comp/dc_test_comp_seq: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
../dc_test_comp/dc_test_comp_seq: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
../dc_test_comp/dc_test_comp_seq: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
../dc_test_comp/dc_test_comp_seq: CMakeFiles/dc_test_comp_seq.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../dc_test_comp/dc_test_comp_seq"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dc_test_comp_seq.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dc_test_comp_seq.dir/build: ../dc_test_comp/dc_test_comp_seq
.PHONY : CMakeFiles/dc_test_comp_seq.dir/build

CMakeFiles/dc_test_comp_seq.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dc_test_comp_seq.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dc_test_comp_seq.dir/clean

CMakeFiles/dc_test_comp_seq.dir/depend:
	cd /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/dc_test_comp_seq.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dc_test_comp_seq.dir/depend

