target_sources(ablateLibrary
        PRIVATE
        mathUtilities.cpp
        demangler.cpp
        petscUtilities.cpp
        petscSupport.cpp
        kokkosUtilities.cpp
        mpiUtilities.cpp

        PUBLIC
        intErrorChecker.hpp
        demangler.hpp
        loggable.hpp
        mathUtilities.hpp
        vectorUtilities.hpp
        petscUtilities.hpp
        petscSupport.hpp
        kokkosUtilities.hpp
        mpiUtilities.hpp
        temporaryWorkingDirectory.hpp
        constants.hpp
        stringUtilities.hpp
        staticInitializer.hpp
        nonCopyable.hpp
        )
