target_sources(ablateLibrary
        PRIVATE
        particleSolver.cpp
        fieldDescription.cpp

        PUBLIC
        field.hpp
        fieldDescription.hpp
        particleSolver.hpp
        )

add_subdirectory(initializers)
add_subdirectory(drag)
add_subdirectory(processes)
add_subdirectory(accessors)

