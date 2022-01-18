#ifndef ABLATELIBRARY_FLOWPROCESS_HPP
#define ABLATELIBRARY_FLOWPROCESS_HPP

#include "process.hpp"

namespace ablate::finiteVolume::processes {

class FlowProcess : public Process {
   public:
    typedef enum { RHO, RHOE, RHOU, RHOV, RHOW } Components;

    /**
     * Private function to decode the euler fields
     * @param flowData
     * @param dim
     * @param conservedValues
     * @param densityYi
     * @param normal
     * @param density
     * @param normalVelocity
     * @param velocity
     * @param internalEnergy
     * @param a
     * @param M
     * @param p
     */
    static void DecodeEulerState(eos::DecodeStateFunction decodeStateFunction, void* decodeStateContext, PetscInt dim, const PetscReal* conservedValues, const PetscReal* densityYi,
                                 const PetscReal* normal, PetscReal* density, PetscReal* normalVelocity, PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* M, PetscReal* p);

    /**
     * Decode the state without a normal direction/velocity
     * @param decodeStateFunction
     * @param decodeStateContext
     * @param dim
     * @param conservedValues
     * @param densityYi
     * @param normal
     * @param density
     * @param normalVelocity
     * @param velocity
     * @param internalEnergy
     * @param a
     * @param M
     * @param p
     */
    static void DecodeEulerState(eos::DecodeStateFunction decodeStateFunction, void* decodeStateContext, PetscInt dim, const PetscReal* conservedValues, const PetscReal* densityYi, PetscReal* density,
                                 PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* M, PetscReal* p);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_FLOWPROCESS_HPP
