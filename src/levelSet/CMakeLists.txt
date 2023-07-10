target_sources(ablateLibrary
        PRIVATE
        LS-VOF.cpp
        levelSetFields.cpp
        levelSetSolver.cpp
        levelSetUtilities.cpp
        cellGrad.cpp

        PUBLIC
        LS-VOF.hpp
        levelSetFields.hpp
        levelSetSolver.hpp
        levelSetUtilities.hpp
        cellGrad.hpp
        )
