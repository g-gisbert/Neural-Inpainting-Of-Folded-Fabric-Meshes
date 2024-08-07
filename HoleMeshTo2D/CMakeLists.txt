cmake_minimum_required(VERSION 3.16.0)
set(CMAKE_CXX_STANDARD 17)

project(pgm LANGUAGES C CXX)

### Configure output locations
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Print the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

### Configure the compiler
# This is a basic, decent setup that should do something sane on most compilers

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    # using Clang (linux or apple) or GCC
    message("Using clang/gcc compiler flags")
    SET(BASE_CXX_FLAGS "-std=c++17 -Wall -Wextra")
    SET(DISABLED_WARNINGS " -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-braces -Wno-unused-private-field")
    SET(TRACE_INCLUDES " -H -Wno-error=unused-command-line-argument")

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        message("Setting clang-specific options")
        SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=3 -fcolor-diagnostics")
        SET(CMAKE_CXX_FLAGS_DEBUG          "-g3 -fsanitize=address -fno-limit-debug-info")
        SET(CMAKE_THREAD_LIBS_INIT "-lpthread")
        SET(CMAKE_HAVE_THREADS_LIBRARY 1)
        SET(CMAKE_USE_WIN32_THREADS_INIT 0)
        SET(CMAKE_USE_PTHREADS_INIT 1)
        SET(THREADS_PREFER_PTHREAD_FLAG ON)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message("Setting gcc-specific options")
        SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
        SET(CMAKE_CXX_FLAGS_DEBUG          "-g3")
        SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable")
    endif()

    SET(GCC_COVERAGE_LINK_FLAGS    "-lm")
    #SET(GCC_COVERAGE_LINK_FLAGS    "-lnlopt -lm")
    SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
    message("Using Visual Studio compiler flags")
    set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /W4")
    set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /MP") # parallel build
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4267\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4244\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4305\"")  # ignore truncation on initialization
    SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

    add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
    add_definitions(-DNOMINMAX)
    add_definitions(-D_USE_MATH_DEFINES)
else()
    # unrecognized
    message( FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]" )
endif()


# == Deps
include_directories(deps/eigen3)
SET(GC_EIGEN_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen3" CACHE INTERNAL "set relative path to eigen" FORCE)

add_subdirectory(deps/geometry-central)
add_subdirectory(deps/polyscope)


# == Build our project stuff
file(GLOB_RECURSE SRCS src/*)
add_executable(pgm "${SRCS}")
target_include_directories(pgm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_include_directories(pgm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/libIGL/include")

# add the args.hxx project which we use for command line args
target_include_directories(pgm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/args")
target_link_libraries(pgm geometry-central polyscope ${CGAL_LIBRARIES} -lm -I./deps/libIGL/include/)
target_precompile_headers(pgm PRIVATE src/pch.h)

