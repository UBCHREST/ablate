target_sources(ablateCore
        PUBLIC
        eos.h
        eos.c
        perfectGas.h
        perfectGas.c
        )

# Allow public access to the header files in the directory
target_include_directories(ablateCore PUBLIC ${CMAKE_CURRENT_LIST_DIR})
