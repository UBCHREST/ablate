target_sources(ablateLibrary
        PUBLIC
        eos.hpp
        perfectGas.hpp
        perfectGas.cpp
        tChem.hpp
        tChem.cpp
        stiffenedGas.hpp
        stiffenedGas.cpp
        )

add_subdirectory(transport)