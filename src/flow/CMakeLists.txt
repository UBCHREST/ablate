
# Include code
target_sources(ablateLibrary
        PUBLIC
        flow.c
        flow.h
        incompressibleFlow.c
        incompressibleFlow.h
#        lowMachFlow.c
#        lowMachFlow.h
        )

# Allow public access to the header files in the directory
target_include_directories(ablateLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR})
