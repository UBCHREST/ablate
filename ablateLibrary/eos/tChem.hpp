#ifndef ABLATECLIENTTEMPLATE_TCHEM_HPP
#define ABLATECLIENTTEMPLATE_TCHEM_HPP

#include "eos.hpp"
#include <filesystem>
#include "utilities/intErrorChecker.hpp"

namespace ablate::eos {

#define CHECKTCHEM(ierr)          if (ierr != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in TChem Library" );

class TChem : public EOS {
   private:
    // hold an error checker for the tchem outside library
    const utilities::IntErrorChecker errorChecker;

    // path to the input files
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;

    // prestore all species
    std::vector<std::string> species;
    int numberSpecies;

    // store a tcWorkingVector
    std::vector<double> workingVector;

    // write/reproduce the periodic table
    static const char* periodicTable;
    inline static const char* periodicTableFileName = "periodictable.dat";

    static PetscErrorCode TChemGasDecodeState(const PetscReal yi[], PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a,
                                                PetscReal* p, void* ctx);
    static PetscErrorCode TChemComputeTemperature(const PetscReal yi[], PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, PetscReal* T, void* ctx);

    // Private static helper functions
    inline const static double TREF = 298.15;
    static int InternalEnergy(int nspec, const PetscReal *yi, double T, double* workArray, double mwMix, double& internalEnergy);
    static PetscErrorCode ComputeTemperature(TChem* tChem, const PetscReal *yi, PetscReal internalEnergyRef, double &T );

   public:
    TChem(std::filesystem::path mechFile, std::filesystem::path thermoFile );
    ~TChem() override;

    // general functions
    void View(std::ostream& stream) const override;

    // species model functions
    const std::vector<std::string>& GetSpecies() const override;

    // EOS functions
    decodeStateFunction GetDecodeStateFunction() override { return TChemGasDecodeState; }
    void* GetDecodeStateContext() override { return this; }
    computeTemperatureFunction GetComputeTemperatureFunction() override { return TChemComputeTemperature; }
    void* GetComputeTemperatureContext() override { return this; }

};

}
#endif  // ABLATECLIENTTEMPLATE_TCHEM_HPP
