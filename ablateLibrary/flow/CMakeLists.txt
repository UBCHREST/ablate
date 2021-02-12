target_sources(ablateLibrary
        PUBLIC
        lowMachFlow.hpp
        lowMachFlow.cpp
        flow.cpp
        flow.hpp
        )

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})
