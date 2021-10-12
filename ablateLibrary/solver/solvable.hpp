#ifndef ABLATELIBRARY_SOLVABLE_HPP
#define ABLATELIBRARY_SOLVABLE_HPP

namespace ablate::solver {
class Solvable {
   public:
    virtual void SetupSolve(TS& timeStepper) = 0;
    virtual Vec GetSolutionVector() = 0;
};
}  // namespace ablate::solve

#endif  // ABLATELIBRARY_SOLVABLE_HPP
