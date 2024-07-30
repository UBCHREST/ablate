#ifndef ABLATELIBRARY_SOOT_HPP
#define ABLATELIBRARY_SOOT_HPP

#include "eos/tChem.hpp"
#include "process.hpp"
#include "utilities/constants.hpp"

namespace ablate::finiteVolume::processes {

class Soot : public Process {
   public:
    //! SolidCarbonDensity
    [[maybe_unused]] inline const static double solidCarbonDensity = 2000;
    //! Scaling term for Ndd going into the Tines ODE Solver
    [[maybe_unused]] inline static double NddScaling = 1e20;
    //! Parameters for Nucleation and Surface Growth Pre Exponentials
    [[maybe_unused]] inline static double nucPreExponential = 1000;
    [[maybe_unused]] inline static double surfPreExponential = 700;
   private:

    // create a separate dm to hold the sources
    DM sourceDm = nullptr;
    // create a separate vec to hold the sources
    Vec sourceVec = nullptr;

    // Petsc options specific to the chemTs. These may be null by default
    PetscOptions petscOptions = nullptr;

    // the eos used to species the species and compute properties
    std::shared_ptr<eos::ChemistryModel> eos;

    // store the default dtInit
    inline const static PetscReal dtInitDefault = 1E-6;

    // store the dtInit, this may be different from default if set with petsc options
    PetscReal dtInit = dtInitDefault;

    // store an optional threshold temperature.  Only compute the reactions if the temperature is above thresholdTemperature
    double thresholdTemperature = 0.0;

    // Store the species in order
    enum OdeSpecies { C_s = 0, H, H2, C2H2, O, O2, OH, CO, TOTAL_ODE_SPECIES };

    // store the name of species used in the ode solver
    inline static const char OdeSpeciesNames[TOTAL_ODE_SPECIES][5] = {"C(S)", "H", "H2", "C2H2", "O", "O2", "OH", "CO"};

    // store the soot ndd ode location
    inline static const PetscInt ODE_NDD = TOTAL_ODE_SPECIES;

    // store the temperature ndd ode location
    inline static const PetscInt ODE_T = TOTAL_ODE_SPECIES + 1;

    // compute the number of ode species
    inline static const PetscInt TotalEquations = TOTAL_ODE_SPECIES + 2;

    // Hold the single point TS
    TS pointTs = nullptr;
    Vec pointData = nullptr;
    Mat pointJacobian = nullptr;

    // Store a struct with the ode point information
    struct OdePointInformation {
        // pass the current density into the ode solver
        PetscReal currentDensity;
        // hold a vector of all yi for scratch to allow
        std::vector<PetscReal> yiScratch;

        // hold a vector of all yi for scratch to allow
        std::vector<PetscReal> speciesSensibleEnthalpyScratch;

        // Hold the function for SpecificHeatConstantVolume
        ablate::eos::ChemistryModel::ThermodynamicTemperatureMassFractionFunction specificHeatConstantVolumeFunction;

        // Hold the function for speciesSensibleEnthalpyFunction
        eos::ChemistryModel::ThermodynamicTemperatureMassFractionFunction speciesSensibleEnthalpyFunction;

        // Precompute the species/T/Ndd offset in the solution vector
        std::array<PetscInt, TotalEquations> speciesOffset;

        // precompute the species index in the species array
        std::array<PetscInt, TOTAL_ODE_SPECIES> speciesIndex;

        // Precompute the enthalpy of formation for each ode species
        std::array<PetscReal, TOTAL_ODE_SPECIES> enthalpyOfFormation;

        // Precompute the mw of each of the ode species
        std::array<PetscReal, TOTAL_ODE_SPECIES> mw;
    };
    OdePointInformation pointInformation;
    /**
     * Private function to integrate single point soot chemistry in time
     * @param ts
     * @param t
     * @param X
     * @param F
     * @param ptr
     * @return
     */
    static PetscErrorCode SinglePointSootChemistryRHS(TS ts, PetscReal t, Vec X, Vec F, void *ptr);

    /**
     * Add the pre computed soot source to the flow
     * @param solver
     * @param dm
     * @param time
     * @param locX
     * @param fVec
     * @param ctx
     * @return
     */
    static PetscErrorCode AddSootChemistrySourceToFlow(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec fVec, void *ctx);

    // Store required soot ode terms
    static constexpr double Ca = 3.;
    static constexpr double Cmin = 700;
    static constexpr double OxidationCollisionEfficiency = .2;
    static constexpr double NdNuclationConversionTerm = 2. / Cmin * 6.02214076e26;

    /**
     * Compute the chemistry for this entire step
     * @param ts
     * @param time
     * @param initialStage
     * @param locX
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeSootChemistryPreStep(FiniteVolumeSolver &, TS ts, PetscReal time, bool initialStage, Vec locX, void *ctx);

    /**
     * Function to compute the soot diameter from mass fraction of carbon and number density
     * @param YCarbon
     * @param Nd
     * @return
     */
    template <typename real>
    static inline double calculateSootDiameter(real YCarbon, real Nd) {
        return std::pow(6 * YCarbon / ablate::utilities::Constants::pi / solidCarbonDensity / (Nd + ablate::utilities::Constants::tiny), 1. / 3.);
    }

    /**
     * Functiont o compute the surface area from mass fraction of carbon and number desnity
     * @param YCarbon
     * @param Nd
     * @param totalDensity
     * @return
     */
    template <typename real>
    static inline real calculateSurfaceArea_V(real YCarbon, real Nd, real totalDensity) {
        real dp = calculateSootDiameter(YCarbon, Nd);
        return ablate::utilities::Constants::pi * dp * dp * totalDensity * Nd;
    }

    template <typename real>
    static inline real calculateNucleationReactionRate(real T, real C2H2Conc, real fv) {
        return nucPreExponential * std::exp(-16103. / T) * C2H2Conc * (1 - fv);
    }

    template <typename real>
    static inline real calculateSurfaceGrowthReactionRate(real T, real C2H2Conc, real SA_V) {
        return surfPreExponential * std::exp(-10064. / T) * C2H2Conc * std::sqrt(SA_V);
    }

    template <typename real>
    static inline double calculateAgglomerationRate(real YCarbon, real Nd, real T, real totalDensity) {
        real dp = calculateSootDiameter(YCarbon, Nd);
        return 2 * Ca * std::sqrt(dp) * std::sqrt(6 * (1.380649e-23) * T / solidCarbonDensity) * (totalDensity * totalDensity * Nd * Nd);  // the Term in the paranthesis is Boltzman's constant
    }

    template <typename real>
    static inline real calculateO2OxidationRate(real YCarbon, real Nd, real O2Conc, real totalDensity, real T, real SA_V) {
        real ka = 200. * std::exp(-15098. / T);
        real kz = 21.3 * std::exp(2063. / T);
        real kb = 4.46e-2 * std::exp(-7650. / T);
        real kT = 1.51e6 * std::exp(-48817. / T);
        real PO2 = O2Conc * (8314.4626) * T / (101325.);  // = |O2| R_u*T in (atm) the first parenthesis term is the UGC, and the second is the conversion of Pascals to atmospheres
        real xA = 1. / (1. + kT / (kb * PO2));
        return (ka * PO2 * xA / (1 + kz * PO2) + kb * PO2 * (1 - xA)) * SA_V;
    }

    template <typename real>
    static inline double calculateOOxidationRate(real OConc, real T, real SA_V, real fv) {
        return 0.001094 * OxidationCollisionEfficiency * std::sqrt(T) * (8314.4626) * OConc * SA_V * (1 - fv);
    }

    template <typename real>
    static inline double calculateOHOxidationRate(real OHConc, real T, real SA_V, real fv) {
        return .001044 * OxidationCollisionEfficiency * std::sqrt(T) * (8314.4626) * OHConc * SA_V * (1 - fv);
    }

   public:
    explicit Soot(const std::shared_ptr<eos::EOS> &eos, const std::shared_ptr<parameters::Parameters> &options = {}, double thresholdTemperature = {}, double surfaceGrowthPreExp = {}, double nucleationPreExp = {});

    ~Soot() override;

    /**
     * Set up mesh dependent initialization
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * Setup up all functions not dependent upon the mesh
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override {}
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_SOOT_HPP
