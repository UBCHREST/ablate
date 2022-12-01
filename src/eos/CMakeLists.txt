target_sources(ablateLibrary
        PRIVATE
        perfectGas.cpp
        stiffenedGas.cpp
        tChem.cpp
        chemTabModel.cpp
        chemistryModel.cpp

        PUBLIC
        eos.hpp
        perfectGas.hpp
        tChem.hpp
        stiffenedGas.hpp
        chemTabModel.hpp
        chemistryModel.hpp
        )

add_subdirectory(transport)
add_subdirectory(tChem)
add_subdirectory(radiationProperties)
