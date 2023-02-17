#include "thermophoreticDiffusion.hpp"
#include "eos/tChemSoot.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::processes::ThermophoreticDiffusion::ThermophoreticDiffusion(std::shared_ptr<eos::EOS> eosIn) : eos(std::move(eosIn)) {}

void ablate::finiteVolume::processes::ThermophoreticDiffusion::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    flow.RegisterRHSFunction(ThermophoreticDiffusionEnergyFlux,
                             &viscosityTemperatureFunction,
                             CompressibleFlowFields::EULER_FIELD,
                             {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD, CompressibleFlowFields::DENSITY_PROGRESS_FIELD},
                             {CompressibleFlowFields::TEMPERATURE_FIELD});

    flow.RegisterRHSFunction(ThermophoreticDiffusionEnergyFlux,
                             &viscosityTemperatureFunction,
                             CompressibleFlowFields::DENSITY_YI_FIELD,
                             {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                             {CompressibleFlowFields::TEMPERATURE_FIELD});
    flow.RegisterRHSFunction(ThermophoreticDiffusionEnergyFlux,
                             &viscosityTemperatureFunction,
                             CompressibleFlowFields::DENSITY_PROGRESS_FIELD,
                             {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_PROGRESS_FIELD},
                             {CompressibleFlowFields::TEMPERATURE_FIELD});
}

PetscErrorCode ablate::finiteVolume::processes::ThermophoreticDiffusion::ThermophoreticDiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                                           const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[],
                                                                                                           const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                                                           PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int EULER_FIELD = 0;
    const int DENSITY_YI_FIELD = 1;
    const int TEMPERATURE_FIELD = 0;

    auto viscosityFunction = (eos::ThermodynamicTemperatureFunction *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[EULER_FIELD] + CompressibleFlowFields::RHO];

    // compute the viscosity on the boundary
    PetscReal temperature = aux[aOff[TEMPERATURE_FIELD]];
    PetscReal mu;
    PetscCall(viscosityFunction->function(field, temperature, &mu, viscosityFunction->context.get()));

    // compute the coefficients for carbon and ndd (note that the negatives cancel).  Note that carbon is always assumed to be at the zero index
    PetscReal coefficient = eos::TChemSoot::ComputeSolidCarbonSensibleEnthalpy(temperature) * field[uOff[DENSITY_YI_FIELD]] * 0.5 * mu / (density * temperature); /*hc*density*Yi*0.5*mu/(rho*T)*/

    // The current energy flux should be zero at the start
    for (PetscInt d = 0; d < dim; ++d) {
        // flux( (carbonCoefficient +  nddCoefficient) * (- dT/dx - dT/dy - dT/dz) . n A
        flux[CompressibleFlowFields::RHOE] += -fg->normal[d] * gradAux[aOff_x[TEMPERATURE_FIELD] + d];
    }
    flux[CompressibleFlowFields::RHOE] *= coefficient;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::ThermophoreticDiffusion::ThermophoreticDiffusionVariableFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                                             const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[],
                                                                                                             const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                                                             PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int EULER_FIELD = 0;
    const int DENSITY_TRANSPORT_FIELD = 1;  // could be densityYi progressYi
    const int TEMPERATURE_FIELD = 0;

    auto viscosityFunction = (eos::ThermodynamicTemperatureFunction *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[EULER_FIELD] + CompressibleFlowFields::RHO];

    // compute the temperature in this volume
    PetscReal temperature = aux[aOff[TEMPERATURE_FIELD]];
    PetscReal mu;
    PetscCall(viscosityFunction->function(field, temperature, &mu, viscosityFunction->context.get()));

    // compute the coefficients for carbon and ndd (note that the negatives cancel).  Note that carbon is always assumed to be at the zero index
    PetscReal coefficient = field[uOff[DENSITY_TRANSPORT_FIELD]] * 0.5 * mu / (density * temperature); /*hc*density*Yi*0.5*mu/(rho*T)*/

    // add in the carbon or ndd flux
    for (PetscInt d = 0; d < dim; ++d) {
        // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
        flux[0] += -fg->normal[d] * coefficient * gradAux[aOff_x[TEMPERATURE_FIELD] + d];
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::ThermophoreticDiffusion,
         "Thermophoretic diffusion the transport of ndd (ThermoPheretic) and solid carbon (ThermoPheretic).", ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"));
