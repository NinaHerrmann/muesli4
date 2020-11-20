# muesli-config.cmake.in

set(MUESLI_VERSION 3.2.1)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was muesli-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

set_and_check(MUESLI_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include/muesli-4.1/include,")

set_and_check(MUESLI_INCLUDE_DETAIL_DIR "${PACKAGE_PREFIX_DIR}/include/muesli-4.1/include,/detail")

set(MUESLI_INCLUDE_DIRS ${MUESLI_INCLUDE_DIR} ${MUESLI_INCLUDE_DETAIL_DIR})

