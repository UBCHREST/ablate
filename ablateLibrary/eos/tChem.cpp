#include "tChem.hpp"

#include <petscts.h>
#include <iostream>
#include <fstream>

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

ablate::eos::TChem::TChem(std::filesystem::path mechFileIn, std::filesystem::path thermoFileIn) : EOS("TChemV1"), errorChecker("Error in TChem library, return code "), mechFile(mechFileIn), thermoFile(thermoFileIn) {
    // TChem requires a periodic file in the working directory.  To simplify setup, we will just write it every time we are run
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> checkMpiError;
    if(rank == 0){
        std::ofstream periodicTableFile(periodicTableFileName);
        periodicTableFile << periodicTable;
        periodicTableFile.close();
    }
    MPI_Barrier(PETSC_COMM_WORLD);

    // initialize TChem (with tabulation off?)
    TC_initChem((char*)mechFile.c_str(), (char*)thermoFile.c_str(), 0, 1.0) >> errorChecker;

    // March over and get each species name
    numberSpecies = TC_getNspec();
    std::vector<char> allSpeciesNames(numberSpecies * LENGTHOFSPECNAME);
    TC_getSnames(numberSpecies, &allSpeciesNames[0]) >> errorChecker;

    // copy each species name
    for(auto s =0; s < numberSpecies; s++){
        auto offset = LENGTHOFSPECNAME * s;
        species.push_back(&allSpeciesNames[offset]);
    }
}

ablate::eos::TChem::~TChem() {
    /* Free memory and reset variables to allow TC_initchem to be called again */
    TC_reset();
}

const std::vector<std::string>& ablate::eos::TChem::GetSpecies() const { return species; }

void ablate::eos::TChem::View(std::ostream& stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tmechFile: " << mechFile<< std::endl;
    stream << "\tthermoFile: " <<thermoFile << std::endl;
}

const char *ablate::eos::TChem::periodicTable = "102 10\n"
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
PetscErrorCode ablate::eos::TChem::DensityGasDecodeState(const PetscReal *yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *velocity, PetscReal *internalEnergy,
                                                         PetscReal *a, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;

    // Get the velocity in this direction to compute the internal energy
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;
    (*internalEnergy) = (totalEnergy)-ke;

    // compute the temperature



//    *p = (parameters->gamma - 1.0) * density * (*internalEnergy);
//    *a = PetscSqrtReal(parameters->gamma * (*p) / density);
    PetscFunctionReturn(0);
}
