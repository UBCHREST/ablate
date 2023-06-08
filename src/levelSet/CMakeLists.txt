target_sources(ablateLibrary
        PRIVATE
        lsSupport.cpp
        LS-VOF.cpp
        levelSetFields.cpp
        levelSetSolver.cpp
        levelSetUtilities.cpp
        cellGrad.cpp

        PUBLIC
        lsSupport.hpp
        LS-VOF.hpp
        levelSetFields.hpp
        levelSetSolver.hpp
        levelSetUtilities.hpp
        cellGrad.hpp
        )
