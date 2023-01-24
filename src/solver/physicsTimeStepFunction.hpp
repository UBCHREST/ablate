#ifndef ABLATELIBRARY_PHYSICSTIMESTEPFUNCTION_HPP
#define ABLATELIBRARY_PHYSICSTIMESTEPFUNCTION_HPP
#include <petsc.h>
#include <map>
namespace ablate::solver {

class PhysicsTimeStepFunction {
   public:
    /**
     * Computes the minimum physics based timestep. Each rank may return a different value, a global reduction will be done.
     */
    virtual double ComputePhysicsTimeStep(TS) = 0;

    /**
     * Computes the individual time steps useful for output/debugging.
     */
    virtual std::map<std::string, double> ComputePhysicsTimeSteps(TS ts) { return {{"", ComputePhysicsTimeStep(ts)}}; }
};

}  // namespace ablate::solver
#endif  // ABLATELIBRARY_PHYSICSTIMESTEPFUNCTION_HPP
