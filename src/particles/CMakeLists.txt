target_sources(ablateLibrary
        PRIVATE
        particleSolver.cpp
        fieldDescription.cpp
        coupledParticleSolver.cpp

        PUBLIC
        field.hpp
        fieldDescription.hpp
        particleSolver.hpp
        coupledParticleSolver.hpp
        )

add_subdirectory(initializers)
add_subdirectory(processes)
add_subdirectory(accessors)

