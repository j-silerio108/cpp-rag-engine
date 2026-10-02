include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(ragengine_tests
    test_similarity.cpp
    test_brute_force_index.cpp
    test_csv_loader.cpp
)
target_link_libraries(ragengine_tests PRIVATE ragengine_core Catch2::Catch2WithMain)
target_compile_definitions(ragengine_tests PRIVATE
    RAGENGINE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(ragengine_tests)
