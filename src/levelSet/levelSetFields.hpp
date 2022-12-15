#ifndef ABLATELIBRARY_LEVELSETFIELD_HPP
#define ABLATELIBRARY_LEVELSETFIELD_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include "domain/fieldDescriptor.hpp"


namespace ablate::levelSet {

class LevelSetFields : public domain::FieldDescriptor {
   private:
    const std::shared_ptr<domain::Region> region;

   public:
    LevelSetFields(std::shared_ptr<domain::Region> region = {});

    // Names of the fieds
    inline const static std::string LEVELSET_FIELD = "levelSet";
    inline const static std::string CURVATURE_FIELD = "curvature";
    inline const static std::string NORMAL_FIELD = "normal";

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
};

//class LevelSetField : public domain::FieldDescriptor {

//  public:

//    // Constructors
//    LevelSetField(std::shared_ptr<domain::Region> = {});
////    LevelSetField(std::shared_ptr<ablate::radialBasis::RBF> rbf = {}, levelSetShape shape = LevelSetField::levelSetShape::SPHERE);

//    // Destructor
//    ~LevelSetField();

//    // Copied from current ABLATE code. Need to talk to Matt M. about how to integrate
//    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;

//   private:


//};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_LEVELSETFIELD_HPP




