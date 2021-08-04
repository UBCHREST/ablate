#include "tChem.hpp"

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
    // initialize TChem (with tabulation off?).  TChem init reads/writes file it can only be done one at a time
    for (int r = 0; r < size; r++) {
        if (r == rank) {
            if(libCount == 0) {
                TC_initChem((char *)mechFile.c_str(), (char *)thermoFile.c_str(), 0, 1.0) >> errorChecker;
            }
            libCount++;

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
    libCount--;

    /* Free memory and reset variables to allow TC_initchem to be called again */
    if(libCount == 0) {
        TC_reset();
    }
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
int ablate::eos::TChem::ComputeSensibleInternalEnergyInternal(int numSpec, double *tempYiWorkingArray, double mwMix, double &internalEnergy) {
    // get the required values
    double totalEnthalpy;
    int err = TC_getMs2HmixMs(tempYiWorkingArray, numSpec + 1, &totalEnthalpy);
    if (err != 0) {
        return err;
    }

    // store the input temperature
    double T = tempYiWorkingArray[0];

    // compute the heat of formation
    tempYiWorkingArray[0] = TREF;
    double enthalpyOfFormation;
    err = TC_getMs2HmixMs(tempYiWorkingArray, numSpec + 1, &enthalpyOfFormation);

    internalEnergy = (totalEnthalpy - enthalpyOfFormation) - T * 1000.0 * RUNIV / mwMix;
    tempYiWorkingArray[0] = T;
    return err;
}

PetscErrorCode ablate::eos::TChem::ComputeTemperatureInternal(int numSpec, double *tempYiWorkingArray, PetscReal internalEnergyRef, double mwMix, double &T) {
    PetscFunctionBeginUser;

    // This is an iterative process to go compute temperature from density
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
                return 0;
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

PetscErrorCode ablate::eos::TChem::TChemComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *massFlux, const PetscReal densityYi[], PetscReal *T, void *ctx) {
    PetscFunctionBeginUser;
    TChem *tChem = (TChem *)ctx;

    // Compute the internal energy from total ener
    // Get the velocity in this direction
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        speedSquare += PetscSqr(massFlux[d] / density);
    }

    // assumed eos
    PetscReal internalEnergyRef = (totalEnergy)-0.5 * speedSquare;

    // Fill the working array
    double *tempYiWorkingArray = &tChem->tempYiWorkingVector[0];
    for (auto sp = 0; sp < tChem->numberSpecies; sp++) {
        tempYiWorkingArray[sp + 1] = densityYi[sp] / density;
    }

    // precompute some values
    double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
    int err = TC_getMs2Wmix(tempYiWorkingArray + 1, tChem->numberSpecies, &mwMix);
    TCCHKERRQ(err);

    // compute the temperature
    PetscErrorCode ierr = ComputeTemperatureInternal(tChem->numberSpecies, tempYiWorkingArray, internalEnergyRef, mwMix, *T);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::TChemGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *velocity, const PetscReal densityYi[], PetscReal *internalEnergy,
                                                       PetscReal *a, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    TChem *tChem = (TChem *)ctx;

    // Get the velocity in this direction to compute the internal energy
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;
    (*internalEnergy) = (totalEnergy)-ke;

    // Fill the working array
    double *tempYiWorkingArray = &tChem->tempYiWorkingVector[0];
    for (auto sp = 0; sp < tChem->numberSpecies; sp++) {
        tempYiWorkingArray[sp + 1] = densityYi[sp] / density;
    }

    // precompute some values
    double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
    int err = TC_getMs2Wmix(tempYiWorkingArray + 1, tChem->numberSpecies, &mwMix);
    TCCHKERRQ(err);

    // compute the temperature
    double temperature;
    PetscErrorCode ierr = ComputeTemperatureInternal(tChem->numberSpecies, tempYiWorkingArray, *internalEnergy, mwMix, temperature);
    CHKERRQ(ierr);

    // compute r
    double R = 1000.0 * RUNIV / mwMix;

    // compute pressure p = rho*R*T
    *p = density * temperature * R;

    // lastly compute the speed of sound
    double cp;
    tempYiWorkingArray[0] = temperature;
    err = TC_getMs2CpMixMs(&tChem->tempYiWorkingVector[0], tChem->numberSpecies + 1, &cp);
    TCCHKERRQ(err);
    double cv = cp - R;
    double gamma = cp / cv;
    *a = PetscSqrtReal(gamma * R * temperature);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::TChemComputeSpeciesSensibleEnthalpy(PetscReal t, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    TChem *tChem = (TChem *)ctx;

    // compute the total enthalpy of each species
    int ierr = TC_getHspecMs(t, tChem->numberSpecies, hi);
    TCCHKERRQ(ierr);

    // subtract away the heat of formation
    for (auto s = 0; s < tChem->numberSpecies; s++) {
        hi[s] -= tChem->speciesHeatOfFormation[s];
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::TChemComputeDensityFunctionFromTemperaturePressure(PetscReal temperature, PetscReal pressure, const PetscReal *yi, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    TChem *tChem = (TChem *)ctx;

    // Fill the working array
    double *tempYiWorkingArray = &tChem->tempYiWorkingVector[0];
    tempYiWorkingArray[0] = temperature;
    for (auto sp = 0; sp < tChem->numberSpecies; sp++) {
        tempYiWorkingArray[sp + 1] = yi[sp];
    }

    // precompute some values
    double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
    int err = TC_getMs2Wmix(tempYiWorkingArray + 1, tChem->numberSpecies, &mwMix);
    TCCHKERRQ(err);

    // compute r
    double R = 1000.0 * RUNIV / mwMix;

    // compute pressure p = rho*R*T
    *density = pressure / (temperature * R);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::TChem::TChemComputeSensibleInternalEnergy(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    TChem *tChem = (TChem *)ctx;

    // Fill the working array
    double *tempYiWorkingArray = &tChem->tempYiWorkingVector[0];
    tempYiWorkingArray[0] = T;
    for (auto sp = 0; sp < tChem->numberSpecies; sp++) {
        tempYiWorkingArray[sp + 1] = yi[sp];
    }

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

#include "parser/registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::TChem, "TChem ideal gas eos", ARG(std::filesystem::path, "mechFile", "the mech file (CHEMKIN Format)"),
         ARG(std::filesystem::path, "thermoFile", "the thermo file (CHEMKIN Format)"));