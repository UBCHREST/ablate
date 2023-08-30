target_sources(ablateLibrary
        PRIVATE
        LS-VOF.cpp
        levelSetUtilities.cpp
        vofMathFunction.cpp
        cellGrad.cpp

        PUBLIC
        LS-VOF.hpp
        levelSetUtilities.hpp
        vofMathFunction.hpp
        cellGrad.hpp
        )
