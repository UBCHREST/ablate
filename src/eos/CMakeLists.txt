target_sources(ablateLibrary
        PRIVATE
        perfectGas.cpp
        stiffenedGas.cpp
        tChem.cpp
        PUBLIC
        eos.hpp
        perfectGas.hpp
        tChem.hpp
        stiffenedGas.hpp
        )

add_subdirectory(transport)
add_subdirectory(tChem)
add_subdirectory(radiationProperties)