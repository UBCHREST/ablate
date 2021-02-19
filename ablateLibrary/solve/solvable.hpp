#ifndef ABLATELIBRARY_SOLVABLE_HPP
#define ABLATELIBRARY_SOLVABLE_HPP

namespace ablate::solve {
class Solvable {
   public:
    virtual Vec SetupSolve(TS& timeStepper) = 0;
};
}  // namespace ablate::solve

#endif  // ABLATELIBRARY_SOLVABLE_HPP
