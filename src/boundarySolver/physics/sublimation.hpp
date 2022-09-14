#ifndef ABLATELIBRARY_SUBLIMATION_HPP
#define ABLATELIBRARY_SUBLIMATION_HPP

#include "boundarySolver/boundaryProcess.hpp"
#include "eos/transport/transportModel.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "finiteVolume/processes/pressureGradientScaling.hpp"
namespace ablate::boundarySolver::physics {

/**
 * produces required source terms in the "gas phase" assuming that the solid phase sublimates and no regression compared to the simulation time
 */
class Sublimation : public BoundaryProcess {
   private:
    const PetscReal latentHeatOfFusion;
    //! transport model used to compute the conductivity
    const std::shared_ptr<ablate::eos::transport::TransportModel> transportModel = nullptr;
    const std::shared_ptr<ablate::eos::EOS> eos;
    const std::shared_ptr<mathFunctions::MathFunction> additionalHeatFlux;
    PetscReal currentTime = 0.0;

    // Store the mass fractions if provided
    const std::shared_ptr<ablate::mathFunctions::FieldFunction> massFractions;
    const mathFunctions::PetscFunction massFractionsFunction;
    void *massFractionsContext;
    PetscInt numberSpecies = 0;

    // toggle to disable any contribution of pressure in the momentum equation
    const bool disablePressure;

    // the pgs is needed for the pressure calculation
    const std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling;

    // store the effectiveConductivity function
    eos::ThermodynamicTemperatureFunction effectiveConductivity;

    // store the function to compute viscosity
    eos::ThermodynamicTemperatureFunction viscosityFunction;

    // reuse fv update temperature function
    eos::ThermodynamicTemperatureFunction computeTemperatureFunction;

    // compute the sensible enthalpy for the blowing term
    eos::ThermodynamicTemperatureFunction computeSensibleEnthalpy;

    // compute the pressure needed for the momentum equation
    eos::ThermodynamicTemperatureFunction computePressure;

    /**
     * Set the species densityYi based upon the blowing rate.  Update the energy if needed to maintain temperature
     */
    void UpdateSpecies(TS ts, ablate::solver::Solver &);

   public:
    explicit Sublimation(PetscReal latentHeatOfFusion, std::shared_ptr<ablate::eos::transport::TransportModel> transportModel, std::shared_ptr<ablate::eos::EOS> eos,
                         const std::shared_ptr<ablate::mathFunctions::FieldFunction> & = {}, std::shared_ptr<mathFunctions::MathFunction> additionalHeatFlux = {},
                         std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling = {}, bool disablePressure = false);

    void Initialize(ablate::boundarySolver::BoundarySolver &bSolver) override;

    /**
     * manual Initialize used for testing
     * @param numberSpecies
     */
    void Initialize(PetscInt numberSpecies);

    /**
     * Support function to compute and insert source terms for this boundary condition
     * @param dim
     * @param fg
     * @param boundaryCell
     * @param uOff
     * @param boundaryValues
     * @param stencilValues
     * @param aOff
     * @param auxValues
     * @param stencilAuxValues
     * @param stencilSize
     * @param stencil
     * @param stencilWeights
     * @param sOff
     * @param source
     * @param ctx
     * @return
     */
    static PetscErrorCode SublimationFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell, const PetscInt uOff[],
                                              const PetscScalar *boundaryValues, const PetscScalar *stencilValues[], const PetscInt aOff[], const PetscScalar *auxValues,
                                              const PetscScalar *stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[],
                                              PetscScalar source[], void *ctx);

    /**
     * Support function to compute output variables
     * @param dim
     * @param fg
     * @param boundaryCell
     * @param uOff
     * @param boundaryValues
     * @param stencilValues
     * @param aOff
     * @param auxValues
     * @param stencilAuxValues
     * @param stencilSize
     * @param stencil
     * @param stencilWeights
     * @param sOff
     * @param source
     * @param ctx
     * @return
     */
    static PetscErrorCode SublimationOutputFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell, const PetscInt uOff[],
                                                    const PetscScalar *boundaryValues, const PetscScalar *stencilValues[], const PetscInt aOff[], const PetscScalar *auxValues,
                                                    const PetscScalar *stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[],
                                                    PetscScalar source[], void *ctx);
};

}  // namespace ablate::boundarySolver::physics

#endif  // ABLATELIBRARY_SUBLIMATION_HPP
