target_sources(ablateLibrary
        PUBLIC
        eos.hpp
        perfectGas.hpp
        perfectGas.cpp
        tChem.hpp
        tChem.cpp
        )

add_subdirectory(transport)