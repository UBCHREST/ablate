target_sources(unitTests
        PRIVATE
        dmPlexTests.cpp
        fieldDescriptionTests.cpp
        )

add_subdirectory(modifiers)
add_subdirectory(RBF)
