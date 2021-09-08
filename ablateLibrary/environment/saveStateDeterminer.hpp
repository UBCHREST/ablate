#ifndef ABLATELIBRARY_SAVESTATEDETERMINER_HPP
#define ABLATELIBRARY_SAVESTATEDETERMINER_HPP

namespace ablate::environment {

class SaveStateDeterminer {
   public:
    /**
     * Determine if the state should be saved at this point
     */
     virtual bool CheckSaveState(TS ts, PetscInt steps, PetscReal time) = 0;
};
}  // namespace ablate::environment


#endif  // ABLATELIBRARY_SAVESTATEDETERMINER_HPP
