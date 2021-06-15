#ifndef ABLATECLIENTTEMPLATE_SPECIESMODEL_HPP
#define ABLATECLIENTTEMPLATE_SPECIESMODEL_HPP

#include <string>
#include <vector>

namespace ablate::solve {

class SpeciesModel {
   public:
    virtual std::vector<std::string> GetSpecies() = 0;
};
}  // namespace ablate::solve
#endif  // ABLATECLIENTTEMPLATE_SPECIESMODEL_HPP
