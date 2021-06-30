#ifndef ABLATECLIENTTEMPLATE_TCHEM_HPP
#define ABLATECLIENTTEMPLATE_TCHEM_HPP

#include <filesystem>
#include "eos.hpp"
#include "utilities/intErrorChecker.hpp"

namespace ablate::eos {

#define CHECKTCHEM(ierr) \
    if (ierr != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in TChem Library");

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

    static PetscErrorCode TChemGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a,
                                              PetscReal* p, void* ctx);
    static PetscErrorCode TChemComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx);

    // Private static helper functions
    inline const static double TREF = 298.15;
    /**
     * the tempYiWorkingArray array is expected to be filled
     * @param numSpec
     * @param tempYiWorkingArray
     * @param T
     * @param mwMix
     * @param internalEnergy
     * @return
     */
    static int InternalEnergy(int numSpec, double* tempYiWorkingArray, double mwMix, double& internalEnergy);

    /**
     * The tempYiWorkingArray is expected to be filled with correct species yi.  The 0 location is set in this function.
     * @param numSpec
     * @param tempYiWorkingArray
     * @param internalEnergyRef
     * @param mwMix
     * @param T
     * @return
     */
    static PetscErrorCode ComputeTemperature(int numSpec, double* tempYiWorkingArray, PetscReal internalEnergyRef, double mwMix, double& T);

   public:
    TChem(std::filesystem::path mechFile, std::filesystem::path thermoFile);
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

}  // namespace ablate::eos
#endif  // ABLATECLIENTTEMPLATE_TCHEM_HPP
