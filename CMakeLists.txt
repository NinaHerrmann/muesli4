cmake_minimum_required (VERSION 3.10.2)
project(Muesli VERSION 4.01)
enable_language(CXX)
message("Muesli Version " ${PROJECT_VERSION})

# required macros
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(MPI REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# config status messages
message(STATUS "Host System: " ${CMAKE_HOST_SYSTEM})
#  -ftree-vectorize
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ftree-vectorize -funroll-loops -fopenmp")

message(STATUS "compiler id: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "compiler flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "compiler flags debug: " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "compiler flags release: " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "compiler flags minsizerel: " ${CMAKE_CXX_FLAGS_MINSIZEREL})
message(STATUS "compiler flags relwithdebinfo: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})

# set default build type, if not specified otherwise
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type: Debug, Release, RelWithDebInfo, or MinSizeRel. " FORCE)
endif(NOT CMAKE_BUILD_TYPE)

message("Build type is: " ${CMAKE_BUILD_TYPE})


message("Muesli installation path: " ${CMAKE_INSTALL_PREFIX})
message("Example build output directory: " ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})


####################################################################################################
# find libs: MPI, OpenMP, and CUDA


### this is a cmake bug: MPI link flags are preceeded by two whitespaces, which leads to one leading whitespace, which is now an error according to policy CMP0004.
string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)
#set(MPI_CXX_COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS} -march=native -ftree-vectorize -funroll-loops -mt_mpi ")

message(STATUS "mpi compiler: " ${MPI_CXX_COMPILER})
message(STATUS "mpi compile flags: " ${MPI_CXX_COMPILE_FLAGS})
message(STATUS "mpi link flags: " ${MPI_CXX_LINK_FLAGS})
message(STATUS "mpi libraries: " ${MPI_CXX_LIBRARIES})
message(STATUS "mpi include paths: " ${MPI_CXX_INCLUDE_PATH})

## OpenMP
find_package(OpenMP)

if (OPENMP_FOUND)
  message(STATUS "OpenMP CXX compiler flags: " ${OpenMP_CXX_FLAGS})
endif (OPENMP_FOUND)

####################################################################################################
# check supported compiler flags

## check supported flags
### common
check_cxx_compiler_flag(-Wall compiler_flag_wall)
check_cxx_compiler_flag(-m64 compiler_flag_m64)
check_cxx_compiler_flag(-fno-strict-aliasing compiler_flag_fno_strict_aliasing)
check_cxx_compiler_flag(-DMPICH_IGNORE_CXX_SEEK compiler_flag_dmpich_ignore_cxx_seek)

### SEQ
## check_cxx_compiler_flag(-Wno-unknown-pragmas compiler_flag_wno_unknown_pragmas)


####################################################################################################
# determine which variants are built

## SEQ
set(BUILD_SEQ true)
message("Build sequential variant.")

## CPU
if (OPENMP_FOUND)
  set(BUILD_CPU true)
  message("OpenMP found can build CPU variant.")
else (OPENMP_FOUND)
  message(WARNING "Do not build CPU variant. OpenMP not found.")
endif (OPENMP_FOUND)
find_package(CUDA)
## GPU
if (CUDA_FOUND)
  set(BUILD_GPU true)
  message("Cuda found can build GPU variant.")
else (CUDA_FOUND)
  message(WARNING "Do not build GPU variant. CUDA not found.")
endif (CUDA_FOUND)
if (MPI_FOUND)
  message("MPI found can use mpirun to start multiple nodes.")
else (MPI_FOUND)
  message(WARNING "Do not use mpirun. MPI not found.")
endif (MPI_FOUND)
####################################################################################################
# add the supported flags to variables

if (compiler_flag_wall)
  set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -Wall)
  set(GPU_COMPILER_OPTIONS_HOST "${GPU_COMPILER_OPTIONS_HOST},\"-Wall\"")
endif (compiler_flag_wall)

if (compiler_flag_m64)
  set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -m64)
  set(GPU_COMPILER_OPTIONS_HOST "${GPU_COMPILER_OPTIONS_HOST},\"-m64\"")
endif (compiler_flag_m64)

if (compiler_flag_fno_strict_aliasing)
  set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -fno-strict-aliasing)
  set(GPU_COMPILER_OPTIONS_HOST "${GPU_COMPILER_OPTIONS_HOST},\"-fno-strict-aliasing\"")
endif (compiler_flag_fno_strict_aliasing)

if (compiler_flag_dmpich_ignore_cxx_seek)
  set(COMMON_COMPILER_OPTIONS ${COMMON_COMPILER_OPTIONS} -DMPICH_IGNORE_CXX_SEEK)
  set(GPU_COMPILER_OPTIONS_HOST "${GPU_COMPILER_OPTIONS_HOST},\"-DMPICH_IGNORE_CXX_SEEK\"")
endif (compiler_flag_dmpich_ignore_cxx_seek)

## SEQ -Wno-unknown-pragmas
if (compiler_flag_wno_unknown_pragmas)
  set(SEQ_COMPILER_OPTIONS ${SEQ_COMPILER_OPTIONS} -Wno-unknown-pragmas)
endif (compiler_flag_wno_unknown_pragmas)

## GPU
if (BUILD_GPU)
  ## include directories
  cuda_include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples/helpers ${PROJECT_SOURCE_DIR}/include/detail ${MPI_CXX_INCLUDE_PATH})

  ## NVCC flags
  set(CUDA_PROPAGATE_HOST_FLAGS "OFF") # these are manually set via -Xcompiler flag ( -Xcompiler "-flag -flag" )


  set(GPU_COMPILER_OPTIONS "-arch=sm_86;-std=c++11;--expt-extended-lambda") # C++11 lambdas are still experimental in CUDA 7.5. This flag may be removed with CUDA 8.0
  set(GPU_COMPILER_OPTIONS_DEBUG "-g;-G;-O0")
  set(GPU_COMPILER_OPTIONS_RELEASE "-O3;-use_fast_math;-Xptxas -O3")
  set(GPU_COMPILER_OPTIONS_MINSIZEREL "-O2")
  set(GPU_COMPILER_OPTIONS_RELWITHDEBINFO "-g;-G;-O2")

  if (OPENMP_FOUND)
    foreach (flag ${OpenMP_CXX_FLAGS})
      string(CONCAT ESCAPED_OPENMP_FLAGS ${ESCAPED_OPENMP_FLAGS} ",\"${flag}\"")
    endforeach (flag ${OpenMP_CXX_FLAGS})
    string(CONCAT GPU_COMPILER_OPTIONS_HOST ${GPU_COMPILER_OPTIONS_HOST} "${ESCAPED_OPENMP_FLAGS}")
  endif (OPENMP_FOUND)

  ## build type flags
  string(REPLACE " -" ";-" CMAKE_CXX_FLAGS_DEBUG_LIST ${CMAKE_CXX_FLAGS_DEBUG})
  foreach (flag ${CMAKE_CXX_FLAGS_DEBUG_LIST})
    string(CONCAT GPU_COMPILER_OPTIONS_HOST_DEBUG ${GPU_COMPILER_OPTIONS_HOST_DEBUG} ",\"${flag}\"")
  endforeach (flag ${CMAKE_CXX_FLAGS_DEBUG_LIST})

  string(REPLACE " -" ";-" CMAKE_CXX_FLAGS_RELEASE_LIST ${CMAKE_CXX_FLAGS_RELEASE})
  foreach (flag ${CMAKE_CXX_FLAGS_RELEASE_LIST})
    string(CONCAT GPU_COMPILER_OPTIONS_HOST_RELEASE ${GPU_COMPILER_OPTIONS_HOST_RELEASE} ",\"${flag}\"")
  endforeach (flag ${CMAKE_CXX_FLAGS_RELEASE_LIST})

  string(REPLACE " -" ";-" CMAKE_CXX_FLAGS_MINSIZEREL_LIST ${CMAKE_CXX_FLAGS_MINSIZEREL})
  foreach (flag ${CMAKE_CXX_FLAGS_MINSIZEREL_LIST})
    string(CONCAT GPU_COMPILER_OPTIONS_HOST_MINSIZEREL ${GPU_COMPILER_OPTIONS_HOST_MINSIZEREL} ",\"${flag}\"")
  endforeach (flag ${CMAKE_CXX_FLAGS_MINSIZEREL_LIST})

  string(REPLACE " -" ";-" CMAKE_CXX_FLAGS_RELWITHDEBINFO_LIST ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  foreach (flag ${CMAKE_CXX_FLAGS_RELWITHDEBINFO_LIST})
    string(CONCAT GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO ${GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO} ",\"${flag}\"")
  endforeach (flag ${CMAKE_CXX_FLAGS_RELWITHDEBINFO_LIST})

  string(CONCAT GPU_HOST_COMPILER_OPTIONS_DEBUG "-Xcompiler " ${GPU_COMPILER_OPTIONS_HOST} ${GPU_COMPILER_OPTIONS_HOST_DEBUG})
  string(CONCAT GPU_HOST_COMPILER_OPTIONS_RELEASE "-Xcompiler " ${GPU_COMPILER_OPTIONS_HOST} ${GPU_COMPILER_OPTIONS_HOST_RELEASE})
  string(CONCAT GPU_HOST_COMPILER_OPTIONS_MINSIZEREL "-Xcompiler " ${GPU_COMPILER_OPTIONS_HOST} ${GPU_COMPILER_OPTIONS_HOST_MINSIZEREL})
  string(CONCAT GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO "-Xcompiler " ${GPU_COMPILER_OPTIONS_HOST} ${GPU_COMPILER_OPTIONS_HOST_RELWITHDEBINFO})

endif (BUILD_GPU)

####################################################################################################
# this custom target is required so that Muesli's header files show up in certain IDEs (such as visual studio)
# if the target is built, nothing actually happens

## find all muesli files
file(GLOB MUESLI_FILES
        ${PROJECT_SOURCE_DIR}/include/*.h
        ${PROJECT_SOURCE_DIR}/include/detail/*.h
        ${PROJECT_SOURCE_DIR}/include/*.cuh
        ${PROJECT_SOURCE_DIR}/include/detail/*.cuh
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cu
        ${PROJECT_SOURCE_DIR}/include/test_hepers/*.cpp
        )

set_source_files_properties(${MUESLI_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
add_custom_target(muesli_ide SOURCES ${MUESLI_FILES})

####################################################################################################
# add muesli libs

set(MUESLI_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/muesli-${PROJECT_VERSION})
set(MUESLI_INCLUDE_INSTALL_DIR ${MUESLI_INSTALL_DIR}/include)
set(MUESLI_INCLUDE_DETAIL_INSTALL_DIR ${MUESLI_INCLUDE_INSTALL_DIR}/detail)

add_library(muesli INTERFACE)
target_include_directories(muesli INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/detail>
        $<BUILD_INTERFACE:${MPI_CXX_INCLUDE_PATH}>
        $<INSTALL_INTERFACE:${MUESLI_INCLUDE_INSTALL_DIR}>
        $<INSTALL_INTERFACE:${MUESLI_INCLUDE_INSTALL_DIR}/detail>
        $<INSTALL_INTERFACE:${MUESLI_INCLUDE_INSTALL_DIR}/test_helpers>
        )
target_compile_options(muesli INTERFACE ${MPI_CXX_COMPILE_FLAGS} ${COMMON_COMPILER_OPTIONS})
target_link_libraries(muesli INTERFACE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
target_compile_features(muesli INTERFACE cxx_lambdas cxx_nullptr cxx_uniform_initialization)

if (BUILD_SEQ)
  add_library(muesli_seq INTERFACE)
  target_compile_options(muesli_seq INTERFACE ${SEQ_COMPILER_OPTIONS})
  target_link_libraries(muesli_seq INTERFACE muesli)
  include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples/helpers ${PROJECT_SOURCE_DIR}/include/detail)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif (BUILD_SEQ)

if (BUILD_CPU)
  add_library(muesli_cpu INTERFACE)
  target_compile_options(muesli_cpu INTERFACE ${OpenMP_CXX_FLAGS} ${CPU_COMPILER_OPTIONS})
  target_link_libraries(muesli_cpu INTERFACE muesli ${OpenMP_CXX_FLAGS})
  include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples/helpers ${PROJECT_SOURCE_DIR}/include/detail)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif (BUILD_CPU)


####################################################################################################
# add install option

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true) #just copy files on install, don't build the examples

# cmake config files
configure_package_config_file(muesli-config.cmake.in
        ${PROJECT_BINARY_DIR}/muesli-config.cmake
        INSTALL_DESTINATION ${MUESLI_INSTALL_DIR}/cmake
        PATH_VARS MUESLI_INCLUDE_INSTALL_DIR MUESLI_INCLUDE_DETAIL_INSTALL_DIR
        NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(
        ${PROJECT_BINARY_DIR}/muesli-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion )

install(FILES ${PROJECT_BINARY_DIR}/muesli-config.cmake
        ${PROJECT_BINARY_DIR}/muesli-config-version.cmake
        DESTINATION ${MUESLI_INSTALL_DIR}/cmake )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src DESTINATION ${MUESLI_INSTALL_DIR})

install(TARGETS muesli EXPORT muesli_targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

if (BUILD_SEQ)
  install(TARGETS muesli_seq EXPORT muesli_targets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          )
endif (BUILD_SEQ)
if (BUILD_CPU)
  install(TARGETS muesli_cpu EXPORT muesli_targets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          )
endif (BUILD_CPU)
install(EXPORT muesli_targets DESTINATION ${MUESLI_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR} NAMESPACE msl:: FILE muesli-lib-config.cmake)

####################################################################################################
# add custom targets. with those it is possible to build all seq, cpu, phi, or gpu targets

if (BUILD_SEQ)
  add_custom_target(seq)
endif (BUILD_SEQ)

if (BUILD_CPU)
  add_custom_target(cpu)
endif (BUILD_CPU)

if (BUILD_GPU)
  add_custom_target(gpu)
endif (BUILD_GPU)
####################################################################################################
# here starts the definition of the executables

## build one executable for each .cpp file in the examples folder
## use of glob is discouraged, so add each file separately
## names must be unique, folders are not considered
set(EXAMPLE_SRCS
        #${PROJECT_SOURCE_DIR}/examples/da_test.cpp
        #${PROJECT_SOURCE_DIR}/examples/dc_map_stencil_test.cpp
        # ${PROJECT_SOURCE_DIR}/examples/dc_test.cpp
        #${PROJECT_SOURCE_DIR}/examples/ds_test.cpp
        #${PROJECT_SOURCE_DIR}/examples/gameoflife.cpp
        #${PROJECT_SOURCE_DIR}/examples/nbody.cpp
        ${PROJECT_SOURCE_DIR}/examples/aco.cpp
        ${PROJECT_SOURCE_DIR}/examples/aco_singlekernel.cpp
        #${PROJECT_SOURCE_DIR}/examples/gaussianblur.cpp
        #${PROJECT_SOURCE_DIR}/examples/gaussianblur-color.cpp
        #${PROJECT_SOURCE_DIR}/examples/jacobisolver.cpp
        #${PROJECT_SOURCE_DIR}/examples/aco.cpp
        #${PROJECT_SOURCE_DIR}/examples/minnested.cpp
        #${PROJECT_SOURCE_DIR}/examples/gassimulation.cpp
        #${PROJECT_SOURCE_DIR}/examples/heatdiffusion.cpp
        #${PROJECT_SOURCE_DIR}/examples/meanblur2.cpp
        #${PROJECT_SOURCE_DIR}/examples/meanblur8.cpp
        #${PROJECT_SOURCE_DIR}/examples/meanblur10.cpp
        #${PROJECT_SOURCE_DIR}/examples/piDA.cpp
        )
foreach (example_src ${EXAMPLE_SRCS})

  ### get name of current example
  get_filename_component(example_name ${example_src} NAME_WE)
  if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${path}/${example_name}")
    file(MAKE_DIRECTORY ${directory} "${PROJECT_SOURCE_DIR}/${path}/${example_name}")
  endif (NOT EXISTS "${PROJECT_SOURCE_DIR}/${path}/${example_name}")
  message(STATUS "${PROJECT_SOURCE_DIR}/${path}/${example_name}")  
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/${path}/${example_name}")

  ### custom target to build all variants for one example
  add_custom_target(${example_name})
  set_target_properties(${example_name} PROPERTIES LINKER_LANGUAGE CXX)
  ### Sequential executable
  if (BUILD_SEQ)
    add_executable(${example_name}_seq ${example_src})
    target_link_libraries(${example_name}_seq ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} muesli_seq)

    ### add to custom targets
    add_dependencies(${example_name} ${example_name}_seq)
    add_dependencies(seq ${example_name}_seq)

  endif (BUILD_SEQ)
  if (BUILD_CPU)
    add_executable(${example_name}_cpu ${example_src})
    target_link_libraries(${example_name}_cpu ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} muesli_cpu)

    ### add to custom targets
    add_dependencies(${example_name} ${example_name}_cpu)
    add_dependencies(cpu ${example_name}_cpu)
  endif (BUILD_CPU)
  if (BUILD_GPU)
    ### TODO: this is not too nice. The source file is a cpp. Therefore, it is build with default c++ compiler; however, we need nvcc. With CUDA_SOURCE_PROPERTY_FORMAT it is possible to compile cpp files with nvcc. However, this breaks the seq, cpu, and phi targets. Therefore, the next lines copy the file to a tmp folder in the build tree, which is used for the gpu target.
    ### possible "solution": should be possible to move this in another CMakeLists.txt file in another folder, then the properties are not valid for the other executables. However, not the whole build in one file anymore.
    get_filename_component(example_name_tmp ${example_src} NAME)
    set(example_src_tmp "tmp/${example_name_tmp}")
    configure_file(${example_src} "${example_src_tmp}")
    set_source_files_properties(${example_src_tmp} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    ## if there is a header file, then copy it as well
    if (EXISTS "${PROJECT_SOURCE_DIR}/examples/${example_name}.h")
      configure_file(${PROJECT_SOURCE_DIR}/examples/${example_name}.h "tmp/${example_name}.h")
    endif (EXISTS "${PROJECT_SOURCE_DIR}/examples/${example_name}.h")

    ## build executable
    cuda_add_executable(${example_name}_gpu ${CMAKE_CURRENT_BINARY_DIR}/${example_src_tmp} OPTIONS ${GPU_COMPILER_OPTIONS} DEBUG ${GPU_COMPILER_OPTIONS_DEBUG} ${GPU_HOST_COMPILER_OPTIONS_DEBUG} RELEASE ${GPU_COMPILER_OPTIONS_RELEASE} ${GPU_HOST_COMPILER_OPTIONS_RELEASE} RELWITHDEBINFO ${GPU_COMPILER_OPTIONS_RELWITHDEBINFO} ${GPU_HOST_COMPILER_OPTIONS_RELWITHDEBINFO} MINSIZEREL ${GPU_COMPILER_OPTIONS_MINSIZEREL} ${GPU_HOST_COMPILER_OPTIONS_MINSIZEREL}) # copy hack: use copied file

    ## link libs
    target_link_libraries(${example_name}_gpu ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} cuda)

    set_property(TARGET ${example_name}_gpu PROPERTY CUDA_ARCHITECTURES 86)

    ### add to custom targets
    add_dependencies(${example_name} ${example_name}_gpu)
    add_dependencies(gpu ${example_name}_gpu)
  endif (BUILD_GPU)
endforeach (example_src ${EXAMPLE_SRCS})
