target_sources(libraryTests
        PRIVATE
        feFlowFunctionSourceTests.cpp
        feFlowDynamicSourceTests.cpp
        )

#add testing support specific to flow cases
add_subdirectory(support)
