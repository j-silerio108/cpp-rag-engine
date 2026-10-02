cmake_minimum_required(VERSION 3.21)
project(RagEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_library(ragengine_core
    src/document.cpp
    src/brute_force_index.cpp
    src/csv_loader.cpp
)
target_include_directories(ragengine_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(ragengine_core PUBLIC -Wall -Wextra -Wpedantic)

add_executable(rag_cli src/main.cpp)
target_link_libraries(rag_cli PRIVATE ragengine_core)

# Static link on MinGW so rag_cli.exe has no runtime dependency on
# libstdc++/libgcc/libwinpthread DLLs being on PATH -- it gets launched by
# whatever process starts the MCP server (mcp_server/server.py), which can't
# be relied on to have the MinGW toolchain's bin directory on its PATH.
if(MINGW)
    target_link_options(rag_cli PRIVATE -static)
endif()

option(RAGENGINE_BUILD_TESTS "Build the unit test suite" ON)
if(RAGENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
