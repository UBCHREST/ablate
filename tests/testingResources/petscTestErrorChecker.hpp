#ifndef ABLATELIBRARY_PETSCTESTERRORCHECKER_HPP
#define ABLATELIBRARY_PETSCTESTERRORCHECKER_HPP

#include <iostream>

class PetscTestErrorChecker {
    friend void operator>>(int ierr, const PetscTestErrorChecker& errorChecker) {
        if (ierr != 0) {
            const char* text;
            char* specific;

            PetscErrorMessage(ierr, &text, &specific);
            std::cerr << text << std::endl << specific << std::endl;
            exit(ierr);
        }
    }
};
#include <gtest/gtest.h>
#include <petscsys.h>
#include <filesystem>
#endif  // ABLATELIBRARY_PETSCTESTERRORCHECKER_HPP
