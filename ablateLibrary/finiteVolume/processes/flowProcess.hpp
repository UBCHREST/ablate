#ifndef ABLATELIBRARY_FLOWPROCESS_HPP
#define ABLATELIBRARY_FLOWPROCESS_HPP

#include "process.hpp"

namespace ablate::finiteVolume::processes {

class FlowProcess : public Process {
   public:
    typedef enum { RHO, RHOE, RHOU, RHOV, RHOW } Components;

    inline const static std::string EULER_FIELD = "euler";
    inline const static std::string DENSITY_YI_FIELD = "densityYi";
    inline const static std::string YI_FIELD = "yi";
    inline const static std::string DENSITY_EV_FIELD = "densityEV";
    inline const static std::string EV_FIELD = "ev";

   public:
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
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_FLOWPROCESS_HPP
