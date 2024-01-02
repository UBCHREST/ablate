#ifndef ABLATELIBRARY_MESHGENERATOR_HPP
#define ABLATELIBRARY_MESHGENERATOR_HPP

#include <array>
#include <memory>
#include <vector>
#include "domain.hpp"
#include "domain/descriptions/meshDescription.hpp"
#include "parameters/parameters.hpp"

namespace ablate::domain {

class MeshGenerator : public Domain {
   private:
    /**
     * Private function to read the desciprtion and generatre the mesh
     * @param name
     * @return
     */
    static DM CreateDM(const std::string& name, std::shared_ptr<ablate::domain::descriptions::MeshDescription> description);

    /**
     * Helper function to replace the dm with a new dm
     * @param originalDm
     * @param replaceDm
     */
    static void ReplaceDm(DM& originalDm, DM& replaceDm);

   public:
    /**
     * Create a mesh using a simple descprtion of nodes/vertcies
     * @param name
     * @param fieldDescriptors
     * @param description
     * @param modifiers
     * @param options
     */
    MeshGenerator(const std::string& name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::shared_ptr<ablate::domain::descriptions::MeshDescription> description,
                  std::vector<std::shared_ptr<modifiers::Modifier>> modifiers, const std::shared_ptr<parameters::Parameters>& options = {});

    ~MeshGenerator() override;
};
}  // namespace ablate::domain
#endif  // ABLATECLIENTTEMPLATE_AXISYMMETRIC_HPP
