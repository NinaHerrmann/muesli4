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
include CMakeFiles/nbody_gpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nbody_gpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nbody_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nbody_gpu.dir/flags.make

CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o: tmp/nbody.cpp
CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o: CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o.depend
CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o: CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o.Debug.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o"
	cd /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_gpu.dir/tmp && /home/n_herr03@WIWI.UNI-MUENSTER.DE/Programme/clion-2022.2.4/bin/cmake/linux/bin/cmake -E make_directory /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_gpu.dir/tmp/.
	cd /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_gpu.dir/tmp && /home/n_herr03@WIWI.UNI-MUENSTER.DE/Programme/clion-2022.2.4/bin/cmake/linux/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_gpu.dir/tmp/./nbody_gpu_generated_nbody.cpp.o -D generated_cubin_file:STRING=/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_gpu.dir/tmp/./nbody_gpu_generated_nbody.cpp.o.cubin.txt -P /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o.Debug.cmake

# Object files for target nbody_gpu
nbody_gpu_OBJECTS =

# External object files for target nbody_gpu
nbody_gpu_EXTERNAL_OBJECTS = \
"/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o"

../nbody/nbody_gpu: CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o
../nbody/nbody_gpu: CMakeFiles/nbody_gpu.dir/build.make
../nbody/nbody_gpu: /usr/local/cuda/lib64/libcudart.so
../nbody/nbody_gpu: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
../nbody/nbody_gpu: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
../nbody/nbody_gpu: CMakeFiles/nbody_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../nbody/nbody_gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nbody_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nbody_gpu.dir/build: ../nbody/nbody_gpu
.PHONY : CMakeFiles/nbody_gpu.dir/build

CMakeFiles/nbody_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nbody_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nbody_gpu.dir/clean

CMakeFiles/nbody_gpu.dir/depend: CMakeFiles/nbody_gpu.dir/tmp/nbody_gpu_generated_nbody.cpp.o
	cd /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug /home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/cmake-build-debug/CMakeFiles/nbody_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nbody_gpu.dir/depend

