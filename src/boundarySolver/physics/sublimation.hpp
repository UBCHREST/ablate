#ifndef ABLATELIBRARY_SUBLIMATION_HPP
#define ABLATELIBRARY_SUBLIMATION_HPP

#include "boundarySolver/boundaryProcess.hpp"
#include "boundarySolver/physics/subModels/sublimationModel.hpp"
#include "eos/radiationProperties/zimmer.hpp"
#include "eos/transport/transportModel.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "finiteVolume/processes/pressureGradientScaling.hpp"
#include "io/interval/interval.hpp"
#include "radiation/surfaceRadiation.hpp"

namespace ablate::boundarySolver::physics {

/**
 * produces required source terms in the "gas phase" assuming that the solid phase sublimates and no regression compared to the simulation time
 */
class Sublimation : public BoundaryProcess, public io::Serializable {
   private:
    // static name of this model
    inline const static std::string sublimationId = "Sublimation";

    //! transport model used to compute the conductivity
    const std::shared_ptr<ablate::eos::transport::TransportModel> transportModel = nullptr;
    const std::shared_ptr<ablate::eos::EOS> eos;
    const std::shared_ptr<mathFunctions::MathFunction> additionalHeatFlux;
    PetscReal currentTime = 0.0;

    //!< Store the mass fractions if provided
    const std::shared_ptr<ablate::mathFunctions::FieldFunction> massFractions;
    const mathFunctions::PetscFunction massFractionsFunction;
    void *massFractionsContext;
    PetscInt numberSpecies = 0;

    /**
     * toggle to disable any contribution of pressure in the momentum equation
     */
    const bool diffusionFlame;

    /**
     * the pgs is needed for the pressure calculation
     */
    const std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling;

    /**
     * the radiation solver for surface heat flux calculation
     */
    std::shared_ptr<ablate::radiation::SurfaceRadiation> radiation;

    /**
     * store the effectiveConductivity function
     */
    eos::ThermodynamicTemperatureFunction effectiveConductivity;

    /**
     * store the function to compute viscosity
     */
    eos::ThermodynamicTemperatureFunction viscosityFunction;

    /**
     * reuse fv update temperature function
     */
    eos::ThermodynamicTemperatureFunction computeTemperatureFunction;

    /**
     * compute the sensible enthalpy for the blowing term
     */
    eos::ThermodynamicTemperatureFunction computeSensibleEnthalpy;

    /**
     * compute the pressure needed for the momentum equation
     */
    eos::ThermodynamicTemperatureFunction computePressure;

    /**
     * interval between the radiation solves
     */
    const std::shared_ptr<io::interval::Interval> radiationInterval;

    /**
     * Emissivity of the fuel surface. For the gray assumption, this indicates how much of the black body radiation is emitted from the fuel surface.
     * */
    const double emissivity;

        /**
     * Absorptivity of the fuel surface. For the gray assumption, this indicates how much of the incoming radiative surface heat flux is absorbed at the fuel surface.
     * */
    const double absorptivity;

    /**
     * Set the species densityYi based upon the blowing rate.  Update the energy if needed to maintain temperature
     */
    void UpdateSpecies(TS ts, ablate::solver::Solver &);

    /**
     * Keep the shared pointer to the solid heat transfer factor is provided
     */
    std::shared_ptr<subModels::SublimationModel> sublimationModel = nullptr;

    // Hold onto a BoundaryPreRHSPointFunctionDefinition to precompute fe heat transfer if returned from the sublimation model
    BoundarySolver::BoundaryPreRHSPointFunctionDefinition solidHeatTransferUpdateFunctionDefinition{};

   public:
    explicit Sublimation(std::shared_ptr<subModels::SublimationModel> sublimationModel, std::shared_ptr<ablate::eos::transport::TransportModel> transportModel, std::shared_ptr<ablate::eos::EOS> eos,
                         const std::shared_ptr<ablate::mathFunctions::FieldFunction> & = {}, std::shared_ptr<mathFunctions::MathFunction> additionalHeatFlux = {},
                         std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling = {}, bool diffusionFlame = false,
                         std::shared_ptr<ablate::radiation::SurfaceRadiation> radiationIn = {}, const std::shared_ptr<io::interval::Interval> &intervalIn = {},
                         const std::shared_ptr<ablate::parameters::Parameters>& ={});

    void Setup(ablate::boundarySolver::BoundarySolver &bSolver) override;
    void Initialize(ablate::boundarySolver::BoundarySolver &bSolver) override;

    /**
     * manual Setup used for testing
     * @param numberSpecies
     */
    void Setup(PetscInt numberSpecies);

    /**
     * only required function, returns the id of the object.  Should be unique for the simulation
     * @return
     */
    [[nodiscard]] const std::string &GetId() const override { return sublimationId; }

    /**
     * assume that the sublimation model does not need to Serialize
     * @return
     */
    [[nodiscard]] SerializerType Serialize() const override { return sublimationModel ? sublimationModel->Serialize() : io::Serializable::SerializerType::none; }

    /**
     * Save the state to the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override {
        PetscFunctionBegin;
        PetscCall(sublimationModel->Save(viewer, sequenceNumber, time));
        PetscFunctionReturn(PETSC_SUCCESS);
    };

    /**
     * Restore the state from the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override {
        PetscFunctionBegin;
        PetscCall(sublimationModel->Restore(viewer, sequenceNumber, time));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    /**
     * Prestep to update the radiation solver
     * @param ts
     * @param solver
     * @return
     */
    static PetscErrorCode SublimationPreRHS(BoundarySolver &, TS ts, PetscReal time, bool initialStage, Vec locX, void *ctx);

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

    /**
     * Call to update the boundary solid model at each point
     * @param time
     * @param dt
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
     * @param ctx
     * @return
     */
    static PetscErrorCode UpdateBoundaryHeatTransferModel(PetscReal time, PetscReal dt, PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg,
                                                          const PetscFVCellGeom *boundaryCell, const PetscInt uOff[], PetscScalar *boundaryValues, const PetscScalar *stencilValues[],
                                                          const PetscInt aOff[], PetscScalar *auxValues, const PetscScalar *stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[],
                                                          const PetscScalar stencilWeights[], void *ctx);
};

}  // namespace ablate::boundarySolver::physics

#endif  // ABLATELIBRARY_SUBLIMATION_HPP
