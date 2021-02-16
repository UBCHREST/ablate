//
// Created by Matt McGurn on 2/15/21.
//

#ifndef ABLATELIBRARY_PETSCERROR_HPP
#define ABLATELIBRARY_PETSCERROR_HPP
#include <petscsys.h>
#include <iostream>

namespace ablate {
namespace utilities {
class PetscError {
    friend void operator>>(PetscErrorCode code,
                       const PetscError& errorChecker)
    {
        std::cout << "ERROR Code: " << code << std::endl;
    }
};
}

inline utilities::PetscError checkError;
}

#endif  // ABLATELIBRARY_PETSCERROR_HPP
