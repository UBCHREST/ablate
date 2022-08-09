#ifndef ABLATELIBRARY_LODIBOUNDARY_HPP
#define ABLATELIBRARY_LODIBOUNDARY_HPP

#include "boundarySolver/boundaryProcess.hpp"
#include "eos/eos.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "finiteVolume/processes/pressureGradientScaling.hpp"

namespace ablate::boundarySolver::lodi {
class LODIBoundary : public BoundaryProcess {
   protected:
    typedef enum {
        RHO = finiteVolume::CompressibleFlowFields::RHO,
        RHOE = finiteVolume::CompressibleFlowFields::RHOE,
        RHOVELN = finiteVolume::CompressibleFlowFields::RHOU,
        RHOVELT1 = finiteVolume::CompressibleFlowFields::RHOV,
        RHOVELT2 = finiteVolume::CompressibleFlowFields::RHOW
    } BoundaryEulerComponents;

    // Global parameters
    const std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling;

    void GetVelAndCPrims(PetscReal velNorm, PetscReal speedOfSound, PetscReal Cp, PetscReal Cv, PetscReal& velNormPrim, PetscReal& speedOfSoundPrim);

    void GetEigenValues(PetscReal veln, PetscReal c, PetscReal velnprm, PetscReal cprm, PetscReal lamda[]) const;

    void GetmdFdn(const PetscInt sOff[], const PetscReal* velNormCord, PetscReal rho, PetscReal T, PetscReal Cp, PetscReal Cv, PetscReal C, PetscReal Enth, PetscReal velnprm, PetscReal Cprm,
                  const PetscReal* Yi, const PetscReal* EV, const PetscReal* sL, const PetscReal transformationMatrix[3][3], PetscReal* mdFdn) const;

    // Compute known/shared values
    PetscInt dims, nEqs, nSpecEqs, nEvEqs, eulerId, speciesId, evId;

    // Keep track of the required fields
    std::vector<std::string> fieldNames;

    // Store eos decode params
    const std::shared_ptr<eos::EOS> eos;
    eos::ThermodynamicFunction computeTemperature;
    eos::ThermodynamicTemperatureFunction computePressureFromTemperature;
    eos::ThermodynamicTemperatureFunction computeSpeedOfSound;
    eos::ThermodynamicTemperatureFunction computeSpecificHeatConstantPressure;
    eos::ThermodynamicTemperatureFunction computeSpecificHeatConstantVolume;
    eos::ThermodynamicTemperatureFunction computeSensibleEnthalpyFunction;
    eos::ThermodynamicFunction computePressure;

   public:
    explicit LODIBoundary(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling = {});

    void Initialize(ablate::boundarySolver::BoundarySolver& bSolver) override;

    /**
     * This function directly sets the known values and is useful for testing
     * @param dims
     * @param nEqs
     * @param nSpecEqs
     * @param nEvEqs
     */
    void Initialize(PetscInt dims, PetscInt nEqs, PetscInt nSpecEqs = 0, PetscInt nEvEqs = 0, const std::vector<domain::Field>& fields = {});

   private:
    eos::ThermodynamicTemperatureFunction computeTemperatureFunction;
};

}  // namespace ablate::boundarySolver::lodi
#endif  // ABLATELIBRARY_LODIBOUNDARY_HPP
