set(target_name "custom-logger")
add_executable(${target_name} ${target_name}.cpp)
target_link_libraries(${target_name} Ginkgo::ginkgo)
target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
