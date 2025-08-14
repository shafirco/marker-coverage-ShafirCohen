enable_testing()

add_executable(mce_tests tests_basic.cpp)
target_link_libraries(mce_tests PRIVATE mce_core)

add_test(NAME mce_basic COMMAND mce_tests)
