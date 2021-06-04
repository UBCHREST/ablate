
# Include code
target_sources(ablateCore
        PUBLIC
        incompressibleFlow.c
        incompressibleFlow.h
        lowMachFlow.c
        lowMachFlow.h
        compressibleFlow.c
        compressibleFlow.h
        fluxDifferencer.h
        fluxDifferencer.c
        fvSupport.h
        fvSupport.c
        flow.c
        flow.h
        )

# Allow public access to the header files in the directory
target_include_directories(ablateCore PUBLIC ${CMAKE_CURRENT_LIST_DIR})
