#include "mixtureFractionCalculator.hpp"
#include "utilities/mathUtilities.hpp"

ablate::chemistry::MixtureFractionCalculator::MixtureFractionCalculator(const std::shared_ptr<ablate::eos::EOS>& eosIn, std::map<std::string, double> massFractionsFuel,
                                                                        std::map<std::string, double> massFractionsOxidizer, const std::vector<std::string>& trackingElementsIn)
    : eos(std::dynamic_pointer_cast<eos::TChem>(eosIn)), trackingElements(trackingElementsIn.empty() ? std::vector<std::string>{"C", "H"} : trackingElementsIn) {
    // make sure that the eos is set
    if (!std::dynamic_pointer_cast<eos::TChem>(eosIn)) {
        throw std::invalid_argument("ablate::chemistry::MixtureFractionCalculator only accepts EOS of type eos::TChem");
    }

    // make sure that the massFractionsOxidizer sums to 1.0
    double massFractionsFuelSum = 0.0;
    for (const auto& [sp, yi] : massFractionsFuel) {
        massFractionsFuelSum += yi;
    }
    if (!ablate::utilities::MathUtilities::Equals(1.0, massFractionsFuelSum)) {
        throw std::invalid_argument("The fuel massFractions (" + std::to_string(massFractionsFuelSum) +
                                    ") provided to ablate::chemistry::MixtureFractionCalculator::MixtureFractionCalculator does not sum to 1.0");
    }

    double massFractionsOxidizerSum = 0.0;
    for (const auto& [sp, yi] : massFractionsOxidizer) {
        massFractionsOxidizerSum += yi;
    }
    if (!ablate::utilities::MathUtilities::Equals(1.0, massFractionsOxidizerSum)) {
        throw std::invalid_argument("The oxidizer massFractions (" + std::to_string(massFractionsOxidizerSum) +
                                    ") provided to ablate::chemistry::MixtureFractionCalculator::MixtureFractionCalculator does not sum to 1.0");
    }

    // first run a sanity check to make sure that the sum of atomic masses for every molecule = MW of this molecule reported back from eos.lookupMW.
    // This check should avoid headaches when mapping species to mixture fraction.
    auto speciesElementInformation = eos->GetSpeciesElementalInformation();
    auto elementInformation = eos->GetElementInformation();
    auto speciesMolecularMass = eos->GetSpeciesMolecularMass();
    for (const auto& [species, info] : speciesElementInformation) {
        double sum = 0.0;
        for (const auto& [e, count] : info) {
            sum += count * elementInformation[e];
        }

        if (!ablate::utilities::MathUtilities::Equals(sum, speciesMolecularMass[species])) {
            throw std::invalid_argument("Problem in ablate::chemistry::MixtureFractionCalculator. Sum of all mixFracMassCoeff =/= 1 for " + species);
        }
    }

    // make sure that each trackingElements is in the elementInformation
    for (const auto& trackingElement : trackingElements) {
        if (!elementInformation.count(trackingElement)) {
            throw std::invalid_argument("The tracking element " + trackingElement + " does not exist in the equation of state");
        }
    }

    // next determine mixFracMassCoeff
    const auto& species = eos->GetSpecies();
    zMixCoefficients.resize(species.size(), 0.0);
    for (std::size_t s = 0; s < species.size(); s++) {
        const auto& speciesName = species[s];
        const auto& speciesElement = speciesElementInformation[speciesName];
        for (const auto& element : trackingElements) {
            zMixCoefficients[s] += speciesElement.at(element) * elementInformation[element] / speciesMolecularMass[speciesName];
        }
    }
    // Compute reference values
    zMixFuel = 0.0;
    zMixOxidizer = 0.0;
    for (std::size_t s = 0; s < species.size(); s++) {
        zMixFuel += zMixCoefficients[s] * massFractionsFuel[species[s]];
        zMixOxidizer += zMixCoefficients[s] * massFractionsOxidizer[species[s]];
    }
}

ablate::chemistry::MixtureFractionCalculator::MixtureFractionCalculator(const std::shared_ptr<ablate::eos::EOS>& eos, const std::shared_ptr<ablate::parameters::Parameters>& massFractionsFuel,
                                                                        const std::shared_ptr<ablate::parameters::Parameters>& massFractionsOxidizer, const std::vector<std::string>& trackingElements)
    : MixtureFractionCalculator(eos, massFractionsFuel->ToMap<double>(), massFractionsOxidizer->ToMap<double>(), trackingElements) {}

#include "registrar.hpp"
REGISTER(ablate::chemistry::MixtureFractionCalculator, ablate::chemistry::MixtureFractionCalculator,
         "Calculate mixture fraction given a list of species using elemental species based on Bilger's (1980) definition of mixture fraction",
         ARG(ablate::eos::EOS, "eos", "The eos with the list of species"), ARG(ablate::parameters::Parameters, "massFractionsFuel", "The initial mass fractions of fuel"),
         ARG(ablate::parameters::Parameters, "massFractionsOxidizer", "The initial mass fractions of oxidizer"),
         OPT(std::vector<std::string>, "trackingElements", "the elements used to element list of the elements you want to track that are in the fuel (defaults to C/H)"));