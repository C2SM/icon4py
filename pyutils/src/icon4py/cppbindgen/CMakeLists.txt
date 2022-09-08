set(CMAKE_CXX_STANDARD 17)

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES CudaIcoCodeGen.cpp Assert.cpp LocToStringUtils.cpp StringUtil.cpp IcoChainSizes.cpp)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(bindings_generator ${SOURCE_FILES})
target_link_libraries(bindings_generator stdc++fs)
