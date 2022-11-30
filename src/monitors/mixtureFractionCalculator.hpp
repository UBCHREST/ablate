#ifndef ABLATELIBRARY_MIXTUREFRACTIONCALCULATOR_HPP
#define ABLATELIBRARY_MIXTUREFRACTIONCALCULATOR_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "eos/tChem.hpp"
#include "mathFunctions/fieldFunction.hpp"

namespace ablate::monitors {

/**
 * Calculate mixture fraction given a list of species using elemental species based on Bilger's (1980) definition of mixture fraction Zeta =
 * (Zi-Zi,ox)/(Zi,f-Zi,ox) which is the normalized elemental mass fraction of any element where f denotes the fuel stream and ox denotes the oxidizer
 * stream.
 */
class MixtureFractionCalculator {
   private:
    //! the reference equation of state
    const std::shared_ptr<ablate::eos::TChem> eos;

    //! the elements used to element list of the elements you want to track that are in the fuel (defaults to C/H)
    const std::vector<std::string> trackingElements;

    //! the mixture fraction of pure fuel and pure oxidizer
    double zMixFuel, zMixOxidizer;

    //! the mixture fraction coefficients for each of the species in the equation of state
    std::vector<double> zMixCoefficients;

    /**
     * static function to help convert from FieldFunction to map
     * @param eos
     * @param massFractionsFuel
     * @return
     */
    static std::map<std::string, double> ToMassFractionMap(const std::shared_ptr<ablate::eos::EOS>& eos, const std::shared_ptr<ablate::mathFunctions::FieldFunction>& massFractions);

   public:
    /**
     * Create the MixtureFractionCalculator to compute the zMixFuel, zMixOxidizer, and trackingElements
     * @param eos The equation of state must be tChem
     * @param massFractionsFuel
     * @param massFractionsOxidizer
     * @param trackingElements defaults to C & H
     */
    MixtureFractionCalculator(const std::shared_ptr<ablate::eos::EOS>& eos, std::map<std::string, double> massFractionsFuel, std::map<std::string, double> massFractionsOxidizer,
                              const std::vector<std::string>& trackingElements = {});

    /**
     * Create the MixtureFractionCalculator to compute the zMixFuel, zMixOxidizer, and trackingElements using field functions.  The point is evaluated at [0, 0, 0] with t = 0;
     * This is used to allow reuse of field function setup in input files
     * @param eos The equation of state must be tChem
     * @param massFractionsFuel
     * @param massFractionsOxidizer
     * @param trackingElements defaults to C & H
     */
    MixtureFractionCalculator(const std::shared_ptr<ablate::eos::EOS>& eos, const std::shared_ptr<ablate::mathFunctions::FieldFunction>& massFractionsFuel,
                              const std::shared_ptr<ablate::mathFunctions::FieldFunction>& massFractionsOxidizer, const std::vector<std::string>& trackingElements = {});
    /**
     * Computes the mixture fraction for a give set of Yi's
     * @tparam T
     * @param yi.  The yi's are assumed to be of length/order of the species in the eos
     * @return
     */
    template <class T>
    inline T Calculate(T* yi) const {
        double zMix = 0.;
        for (std::size_t s = 0; s < zMixCoefficients.size(); s++) {
            zMix += zMixCoefficients[s] * yi[s];
        }
        return (zMix - zMixOxidizer) / (zMixFuel - zMixOxidizer);
    }

    /**
     * Return accesses the base eos
     * @return
     */
    std::shared_ptr<ablate::eos::TChem> GetEos() { return eos; }
};

}  // namespace ablate::chemistry
#endif  // ABLATELIBRARY_MIXTUREFRACTIONCALCULATOR_HPP
