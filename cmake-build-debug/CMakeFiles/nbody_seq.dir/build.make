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
include CMakeFiles/nbody_seq.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nbody_seq.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nbody_seq.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nbody_seq.dir/flags.make

CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o: CMakeFiles/nbody_seq.dir/flags.make
CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o: ../examples/nbody.cpp
CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o: CMakeFiles/nbody_seq.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o -MF CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o.d -o CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o -c /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/examples/nbody.cpp

CMakeFiles/nbody_seq.dir/examples/nbody.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nbody_seq.dir/examples/nbody.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/examples/nbody.cpp > CMakeFiles/nbody_seq.dir/examples/nbody.cpp.i

CMakeFiles/nbody_seq.dir/examples/nbody.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nbody_seq.dir/examples/nbody.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/examples/nbody.cpp -o CMakeFiles/nbody_seq.dir/examples/nbody.cpp.s

# Object files for target nbody_seq
nbody_seq_OBJECTS = \
"CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o"

# External object files for target nbody_seq
nbody_seq_EXTERNAL_OBJECTS =

../nbody/nbody_seq: CMakeFiles/nbody_seq.dir/examples/nbody.cpp.o
../nbody/nbody_seq: CMakeFiles/nbody_seq.dir/build.make
../nbody/nbody_seq: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
../nbody/nbody_seq: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
../nbody/nbody_seq: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
../nbody/nbody_seq: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
../nbody/nbody_seq: CMakeFiles/nbody_seq.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../nbody/nbody_seq"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nbody_seq.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nbody_seq.dir/build: ../nbody/nbody_seq
.PHONY : CMakeFiles/nbody_seq.dir/build

CMakeFiles/nbody_seq.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nbody_seq.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nbody_seq.dir/clean

CMakeFiles/nbody_seq.dir/depend:
	cd /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_seq.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nbody_seq.dir/depend

