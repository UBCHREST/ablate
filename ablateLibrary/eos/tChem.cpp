#include "tChem.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

#include <fstream>
#include <iostream>

#if defined(PETSC_HAVE_TCHEM)
#if defined(MAX)
#undef MAX
#endif
#if defined(MIN)
#undef MIN
#endif
#include <TC_interface.h>
#include <TC_params.h>
#include <utilities/mpiError.hpp>
#else
#error TChem is required.  Reconfigure PETSc using --download-tchem.
#endif

ablate::eos::TChem::TChem(std::filesystem::path mechFileIn, std::filesystem::path thermoFileIn)
    : EOS("TChemV1"), errorChecker("Error in TChem library, return code "), mechFile(mechFileIn), thermoFile(thermoFileIn) {
    // TChem requires a periodic file in the working directory.  To simplify setup, we will just write it every time we are run
    int size = 1;
    int rank = 0;

    int mpiInitialized;
    MPI_Initialized(&mpiInitialized) >> checkMpiError;
    if (mpiInitialized) {
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> checkMpiError;
        if (rank == 0) {
            std::ofstream periodicTableFile(periodicTableFileName);
            periodicTableFile << periodicTable;
            periodicTableFile.close();
        }
        MPI_Barrier(PETSC_COMM_WORLD);
        MPI_Comm_size(PETSC_COMM_WORLD, &size) >> checkMpiError;
    }

    if (libUsed) {
        throw std::runtime_error("Only a single instance of the TChem EOS can be instance at a time");
    }
    libUsed = true;

    // initialize TChem (with tabulation off?).  TChem init reads/writes file it can only be done one at a time
    for (int r = 0; r < size; r++) {
        if (r == rank) {
            TC_initChem((char *)mechFile.c_str(), (char *)thermoFile.c_str(), 0, 1.0) >> errorChecker;

            // Perform the local init
            // March over and get each species name
            numberSpecies = TC_getNspec();
            std::vector<char> allSpeciesNames(numberSpecies * LENGTHOFSPECNAME);
            TC_getSnames(numberSpecies, &allSpeciesNames[0]) >> errorChecker;

            // copy each species name
            for (auto s = 0; s < numberSpecies; s++) {
                auto offset = LENGTHOFSPECNAME * s;
                species.push_back(&allSpeciesNames[offset]);
            }

            // size the working vector
            tempYiWorkingVector.resize(numberSpecies + 1);
            sourceWorkingVector.resize(numberSpecies + 1);

            // precompute the speciesHeatOfFormation at tref
            speciesHeatOfFormation.resize(numberSpecies);
            TC_getHspecMs(TREF, numberSpecies, &speciesHeatOfFormation[0]) >> errorChecker;
        }
        if (mpiInitialized) {
            MPI_Barrier(PETSC_COMM_WORLD);
        }
    }
}

ablate::eos::TChem::~TChem() {
    /* Free memory and reset variables to allow TC_initchem to be called again */
    if (libUsed) {
        TC_reset();
    }
    libUsed = false;
}

const std::vector<std::string> &ablate::eos::TChem::GetSpecies() const { return species; }

void ablate::eos::TChem::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tmechFile: " << mechFile << std::endl;
    stream << "\tthermoFile: " << thermoFile << std::endl;
}

/**
 * the tempYiWorkingArray array is expected to be filled.
 * @param yi
 * @param T
 * @return
 */
int ablate::eos::TChem::ComputeEnthalpyOfFormation(int numSpec, double *tempYiWorkingArray, double &enthalpyOfFormation) {
    // compute the heat of formation
    double currentT = tempYiWorkingArray[0];
    tempYiWorkingArray[0] = TREF;
    int err = TC_getMs2HmixMs(tempYiWorkingArray, numSpec + 1, &enthalpyOfFormation);
    tempYiWorkingArray[0] = currentT;
    return err;
}

/**
 * the tempYiWorkingArray array is expected to be filled.
 * @param yi
 * @param T
 * @return
 */
int ablate::eos::TChem::ComputeSensibleInternalEnergyInternal(int numSpec, double *tempYiWorkingArray, double mwMix, double &internalEnergy) {
    // get the required values
    double totalEnthalpy;
    int err = TC_getMs2HmixMs(tempYiWorkingArray, numSpec + 1, &totalEnthalpy);
    if (err != 0) {
        return err;
    }

    // compute the heat of formation
    double enthalpyOfFormation;
    err = ComputeEnthalpyOfFormation(numSpec, tempYiWorkingArray, enthalpyOfFormation);

    internalEnergy = (totalEnthalpy - enthalpyOfFormation) - tempYiWorkingArray[0] * 1000.0 * RUNIV / mwMix;
    return err;
}

PetscErrorCode ablate::eos::TChem::ComputeTemperatureInternal(int numSpec, double *tempYiWorkingArray, PetscReal internalEnergyRef, double mwMix, double &T) {
    PetscFunctionBeginUser;

    // This is an iterative process to go compute temperature from density
    // TODO: update to use temperature guess
    double t2 = 300.0;

    // set some constants
    const auto EPS_T_RHO_E = 1E-8;
    const auto ITERMAX_T = 100;

    // compute the first error
    double e2;
    tempYiWorkingArray[0] = t2;

    int err = ComputeSensibleInternalEnergyInternal(numSpec, tempYiWorkingArray, mwMix, e2);
    TCCHKERRQ(err);
    double f2 = internalEnergyRef - e2;
    T = t2;  // set for first guess
    if (PetscAbs(f2) > EPS_T_RHO_E) {
        double t0 = t2;
        double f0 = f2;
        double t1 = t0 + 1;
        double e1;
        tempYiWorkingArray[0] = t1;
        err = ComputeSensibleInternalEnergyInternal(numSpec, tempYiWorkingArray, mwMix, e1);
        TCCHKERRQ(err);
        double f1 = internalEnergyRef - e1;

        for (int it = 0; it < ITERMAX_T; it++) {
            t2 = t1 - f1 * (t1 - t0) / (f1 - f0 + 1E-30);
            t2 = PetscMax(1.0, t2);
            tempYiWorkingArray[0] = t2;
            err = ComputeSensibleInternalEnergyInternal(numSpec, tempYiWorkingArray, mwMix, e2);
            TCCHKERRQ(err);
            f2 = internalEnergyRef - e2;
            if (PetscAbs(f2) <= EPS_T_RHO_E) {
                T = t2;
                PetscFunctionReturn(0);
            }
            t0 = t1;
            t1 = t2;
            f0 = f1;
            f1 = f2;
        }
        T = t2;
    }
    PetscFunctionReturn(0);
}

void ablate::eos::TChem::FillWorkingVectorFromMassFractions(int numSpec, double temperature, const double *yi, double *workingVector) {
    workingVector[0] = temperature;
    PetscScalar yiSum = 0.0;
    for (PetscInt sp = 0; sp < numSpec - 1; sp++) {
        // Limit the bounds
        workingVector[sp + 1] = PetscMax(0.0, yi[sp]);
        workingVector[sp + 1] = PetscMin(1.0, workingVector[sp + 1]);
        yiSum += workingVector[sp + 1];
    }
    if (yiSum > 1.0) {
        for (PetscInt sp = 0; sp < numSpec - 1; sp++) {
            // Limit the bounds
            workingVector[sp + 1] /= yiSum;
        }
        workingVector[numSpec] = 0.0;
    } else {
        workingVector[numSpec] = 1.0 - yiSum;
    }
}

/**
 * Fill and Normalize the density species mass fractions
 * @param numSpec
 * @param yi
 */
void ablate::eos::TChem::FillWorkingVectorFromDensityMassFractions(int numSpec, double density, double temperature, const double *densityYi, double *workingVector) {
    workingVector[0] = temperature;
    PetscScalar yiSum = 0.0;
    for (PetscInt sp = 0; sp < numSpec - 1; sp++) {
        // Limit the bounds
        workingVector[sp + 1] = PetscMax(0.0, densityYi[sp] / density);
        workingVector[sp + 1] = PetscMin(1.0, workingVector[sp + 1]);
        yiSum += workingVector[sp + 1];
    }
    if (yiSum > 1.0) {
        for (PetscInt sp = 0; sp < numSpec - 1; sp++) {
            // Limit the bounds
            workingVector[sp + 1] /= yiSum;
        }
        workingVector[numSpec] = 0.0;
    } else {
        workingVector[numSpec] = 1.0 - yiSum;
    }
}

ablate::eos::ThermodynamicFunction ablate::eos::TChem::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::PerfectGas requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::PerfectGas requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD Field");
    }

    return ThermodynamicFunction{.function = thermodynamicFunctions.at(property).first,
                                 .context = std::make_shared<FunctionContext>(
                                     FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .densityYiOffset = densityYiField->offset, .tChem = this})};
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::TChem::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::PerfectGas requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
    if (densityYiField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::PerfectGas requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD Field");
    }

    return ThermodynamicTemperatureFunction{.function = thermodynamicFunctions.at(property).second,
                                            .context = std::make_shared<FunctionContext>(
                                                FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .densityYiOffset = densityYiField->offset, .tChem = this})};
}

PetscErrorCode ablate::eos::TChem::TemperatureFunction(const PetscReal *conserved, PetscReal *property, void *ctx) { return TemperatureTemperatureFunction(conserved, 300, property, ctx); }
PetscErrorCode ablate::eos::TChem::TemperatureTemperatureFunction(const PetscReal *conserved, PetscReal temperatureGuess, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    auto tChem = functionContext->tChem;
    // Compute the internal energy from total ener
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    // assumed eos
    PetscReal internalEnergyRef = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;

    // Fill the working array
    auto tempYiWorkingArray = tChem->tempYiWorkingVector.data();
    FillWorkingVectorFromDensityMassFractions(tChem->numberSpecies, density, temperatureGuess, conserved + functionContext->densityYiOffset, tempYiWorkingArray);

    // precompute some values
    double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
    int err = TC_getMs2Wmix(tempYiWorkingArray + 1, tChem->numberSpecies, &mwMix);
    TCCHKERRQ(err);

    // compute the temperature
    PetscErrorCode ierr = ComputeTemperatureInternal(tChem->numberSpecies, tempYiWorkingArray, internalEnergyRef, mwMix, *temperature);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::PressureFunction(const PetscReal *conserved, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = PressureTemperatureFunction(conserved, temperature, pressure, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::PressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    auto tChem = functionContext->tChem;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    auto tempYiWorkingArray = tChem->tempYiWorkingVector.data();
    FillWorkingVectorFromDensityMassFractions(tChem->numberSpecies, density, temperature, conserved + functionContext->densityYiOffset, tempYiWorkingArray);

    // precompute some values
    double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
    int err = TC_getMs2Wmix(tempYiWorkingArray + 1, tChem->numberSpecies, &mwMix);
    TCCHKERRQ(err);

    // compute r
    double R = 1000.0 * RUNIV / mwMix;

    // compute pressure p = rho*R*T
    *pressure = density * temperature * R;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    *sensibleInternalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::InternalSensibleEnergyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    auto tChem = functionContext->tChem;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    auto tempYiWorkingArray = tChem->tempYiWorkingVector.data();
    FillWorkingVectorFromDensityMassFractions(tChem->numberSpecies, density, temperature, conserved + functionContext->densityYiOffset, tempYiWorkingArray);

    // precompute some values
    double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
    int err = TC_getMs2Wmix(tempYiWorkingArray + 1, tChem->numberSpecies, &mwMix);
    TCCHKERRQ(err);

    // compute the sensibleInternalEnergy
    double sensibleInternalEnergyCompute = 0;
    err = ComputeSensibleInternalEnergyInternal(tChem->numberSpecies, tempYiWorkingArray, mwMix, sensibleInternalEnergyCompute);
    *sensibleInternalEnergy = sensibleInternalEnergyCompute;
    TCCHKERRQ(err);

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SensibleEnthalpyTemperatureFunction(conserved, temperature, sensibleEnthalpy, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    auto tChem = functionContext->tChem;

    // Fill the working array
    auto tempYiWorkingArray = tChem->tempYiWorkingVector.data();
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    FillWorkingVectorFromDensityMassFractions(tChem->numberSpecies, density, temperature, conserved + functionContext->densityYiOffset, tempYiWorkingArray);

    // get the required values
    double totalEnthalpy;
    int err = TC_getMs2HmixMs(tempYiWorkingArray, tChem->numberSpecies + 1, &totalEnthalpy);
    if (err != 0) {
        return err;
    }

    // compute the heat of formation
    double enthalpyOfFormation;
    err = ComputeEnthalpyOfFormation(tChem->numberSpecies, tempYiWorkingArray, enthalpyOfFormation);
    CHKERRQ(err);

    *sensibleEnthalpy = totalEnthalpy - enthalpyOfFormation;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpecificHeatConstantVolumeTemperatureFunction(conserved, temperature, cv, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    auto tChem = functionContext->tChem;

    // Fill the working array
    auto tempYiWorkingArray = tChem->tempYiWorkingVector.data();
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    FillWorkingVectorFromDensityMassFractions(tChem->numberSpecies, density, temperature, conserved + functionContext->densityYiOffset, tempYiWorkingArray);

    // call the tChem library
    int err = TC_getMs2CvMixMs(tempYiWorkingArray, tChem->numberSpecies + 1, cv);
    TCCHKERRQ(err);

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpecificHeatConstantPressureTemperatureFunction(conserved, temperature, cp, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    auto tChem = functionContext->tChem;

    // Fill the working array
    auto tempYiWorkingArray = tChem->tempYiWorkingVector.data();
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    FillWorkingVectorFromDensityMassFractions(tChem->numberSpecies, density, temperature, conserved + functionContext->densityYiOffset, tempYiWorkingArray);

    // call the tChem library
    int err = TC_getMs2CpMixMs(tempYiWorkingArray, tChem->numberSpecies + 1, cp);
    TCCHKERRQ(err);

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpeedOfSoundTemperatureFunction(conserved, temperature, speedOfSound, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    auto tChem = functionContext->tChem;

    // Fill the working array
    auto tempYiWorkingArray = tChem->tempYiWorkingVector.data();
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    FillWorkingVectorFromDensityMassFractions(tChem->numberSpecies, density, temperature, conserved + functionContext->densityYiOffset, tempYiWorkingArray);

    // precompute some values
    double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
    int err = TC_getMs2Wmix(tempYiWorkingArray + 1, tChem->numberSpecies, &mwMix);
    TCCHKERRQ(err);

    // compute r
    double R = 1000.0 * RUNIV / mwMix;

    // lastly compute the speed of sound
    double cp;
    err = TC_getMs2CpMixMs(&tChem->tempYiWorkingVector[0], tChem->numberSpecies + 1, &cp);
    TCCHKERRQ(err);
    double cv = cp - R;
    double gamma = cp / cv;
    *speedOfSound = PetscSqrtReal(gamma * R * temperature);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscErrorCode ierr = TemperatureFunction(conserved, &temperature, ctx);
    CHKERRQ(ierr);
    ierr = SpeciesSensibleEnthalpyTemperatureFunction(conserved, temperature, hi, ctx);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    auto tChem = functionContext->tChem;

    // compute the total enthalpy of each species
    int ierr = TC_getHspecMs(temperature, tChem->numberSpecies, hi);
    TCCHKERRQ(ierr);

    // subtract away the heat of formation
    for (auto s = 0; s < tChem->numberSpecies; s++) {
        hi[s] -= tChem->speciesHeatOfFormation[s];
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::TChem::DensityTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
ablate::eos::FieldFunction ablate::eos::TChem::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2) const {
    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                // Fill the working array
                auto tempYiWorkingArray = tempYiWorkingVector.data();
                FillWorkingVectorFromMassFractions(numberSpecies, temperature, yi, tempYiWorkingArray);

                // precompute some values
                double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
                TC_getMs2Wmix(tempYiWorkingArray + 1, numberSpecies, &mwMix) >> errorChecker;

                // compute r
                double R = 1000.0 * RUNIV / mwMix;

                // compute pressure p = rho*R*T
                PetscReal density = pressure / (temperature * R);

                PetscReal sensibleInternalEnergy;
                ComputeSensibleInternalEnergyInternal(numberSpecies, tempYiWorkingArray, mwMix, sensibleInternalEnergy) >> errorChecker;

                // convert to total sensibleEnergy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;

                conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + sensibleInternalEnergy);
                for (PetscInt d = 0; d < dim; d++) {
                    conserved[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density * velocity[d];
                }
            };
            if (property1 == ThermodynamicProperty::Temperature) {
                return tp;
            } else {
                return [tp](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    tp(temperature, pressure, dim, velocity, yi, conserved);
                };
            }
        }
        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::PerfectGas.");

    } else if (finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD == field) {
        if (property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) {
            return [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                auto tempYiWorkingArray = tempYiWorkingVector.data();
                FillWorkingVectorFromMassFractions(numberSpecies, temperature, yi, tempYiWorkingArray);

                // precompute some values
                double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
                TC_getMs2Wmix(tempYiWorkingArray + 1, numberSpecies, &mwMix) >> errorChecker;

                // compute r
                double R = 1000.0 * RUNIV / mwMix;

                // compute pressure p = rho*R*T
                PetscReal density = pressure / (temperature * R);

                for (PetscInt c = 0; c < numberSpecies; c++) {
                    conserved[c] = density * yi[c];
                }
            };
        } else if (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature) {
            return [this](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                auto tempYiWorkingArray = tempYiWorkingVector.data();
                FillWorkingVectorFromMassFractions(numberSpecies, temperature, yi, tempYiWorkingArray);

                // precompute some values
                double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
                TC_getMs2Wmix(tempYiWorkingArray + 1, numberSpecies, &mwMix) >> errorChecker;

                // compute r
                double R = 1000.0 * RUNIV / mwMix;

                // compute pressure p = rho*R*T
                PetscReal density = pressure / (temperature * R);

                for (PetscInt c = 0; c < numberSpecies; c++) {
                    conserved[c] = density * yi[c];
                }
            };
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::PerfectGas.");
    } else {
        throw std::invalid_argument("Unknown field type " + field + " for ablate::eos::PerfectGas.");
    }
}

const char *ablate::eos::TChem::periodicTable =
    "102 10\n"
    "H          HE         LI         BE         B          C          N          O          F          NE\n"
    "  1.00797    4.00260    6.93900    9.01220   10.81100   12.01115   14.00670   15.99940   18.99840   20.18300\n"
    "NA         MG         AL         SI         P          S          CL         AR         K          CA\n"
    " 22.98980   24.31200   26.98150   28.08600   30.97380   32.06400   35.45300   39.94800   39.10200   40.08000\n"
    "SC         TI         V          CR         MN         FE         CO         NI         CU         ZN\n"
    " 44.95600   47.90000   50.94200   51.99600   54.93800   55.84700   58.93320   58.71000   63.54000   65.37000\n"
    "GA         GE         AS         SE         BR         KR         RB         SR         Y          ZR\n"
    " 69.72000   72.59000   74.92160   78.96000   79.90090   83.80000   85.47000   87.62000   88.90500   91.22000\n"
    "NB         MO         TC         RU         RH         PD         AG         CD         IN         SN\n"
    " 92.90600   95.94000   99.00000  101.07000  102.90500  106.40000  107.87000  112.40000  114.82000  118.69000\n"
    "SB         TE         I          XE         CS         BA         LA         CE         PR         ND\n"
    "121.75000  127.60000  126.90440  131.30000  132.90500  137.34000  138.91000  140.12000  140.90700  144.24000\n"
    "PM         SM         EU         GD         TB         DY         HO         ER         TM         YB\n"
    "145.00000  150.35000  151.96000  157.25000  158.92400  162.50000  164.93000  167.26000  168.93400  173.04000\n"
    "LU         HF         TA         W          RE         OS         IR         PT         AU         HG\n"
    "174.99700  178.49000  180.94800  183.85000  186.20000  190.20000  192.20000  195.09000  196.96700  200.59000\n"
    "TL         PB         BI         PO         AT         RN         FR         RA         AC         TH\n"
    "204.37000  207.19000  208.98000  210.00000  210.00000  222.00000  223.00000  226.00000  227.00000  232.03800\n"
    "PA         U          NP         PU         AM         CM         BK         CF         ES         FM\n"
    "231.00000  238.03000  237.00000  242.00000  243.00000  247.00000  249.00000  251.00000  254.00000  253.00000\n"
    "D          E\n"
    "  2.01410    5.45E-4  \n"
    "";

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::TChem, "TChem ideal gas eos", ARG(std::filesystem::path, "mechFile", "the mech file (CHEMKIN Format)"),
         ARG(std::filesystem::path, "thermoFile", "the thermo file (CHEMKIN Format)"));