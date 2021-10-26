#ifndef ABLATELIBRARY_SUBDOMAIN_HPP
#define ABLATELIBRARY_SUBDOMAIN_HPP
#include <petsc.h>
#include <map>
#include <mathFunctions/fieldFunction.hpp>
#include <memory>
#include <string>
#include "domain.hpp"
#include "fieldDescriptor.hpp"

namespace ablate::domain {

class SubDomain {
   private:
    // const pointer to the parent domain/dm
    const std::weak_ptr<Domain> domain;

    // store the region used to define this subDomain
    const std::shared_ptr<domain::Region> region;

    // Keep a name for this subDomain for output/debug
    std::string name;

    // The label used to describe this subDomain
    DMLabel label;

    // Keep track of fields that live in this subDomain;
    std::map<std::string, Field> fieldsByName;

    // also keep the fields by type for faster iteration
    std::map<FieldType, std::vector<Field>> fieldsByType;

    // Each subDomain will operate over a ds
    PetscDS discreteSystem;
    PetscDS auxDiscreteSystem;

    // The auxDm and auxVec are for only this subDomain
    DM auxDM;
    Vec auxVec;

    // Create/store a subDM for output
    DM subDM;
    Vec subSolutionVec;
    DM subAuxDM;
    Vec subAuxVec;

    // support call to copy from global to sub vec
    void CopyGlobalToSubVector(DM subDM, DM gDM, Vec subVec, Vec globVec, const std::vector<Field>& subFields, const std::vector<Field>& gFields = {}, bool localVector = false);

   public:
    SubDomain(std::weak_ptr<Domain> domain, std::shared_ptr<domain::Region>);
    ~SubDomain();

    Field RegisterField(const FieldDescriptor& fieldDescriptor, PetscObject field);

    /**
     * Create the subDomain discrete system
     */
    void InitializeDiscreteSystem();

    /**
     * Create the auxDM, auxVec, and other structures on the subDomain
     */
    void CreateSubDomainStructures();

    // Returns the local field information
    inline const Field& GetField(const std::string& fieldName) const {
        if (fieldsByName.count(fieldName)) {
            return fieldsByName.at(fieldName);
        } else {
            throw std::invalid_argument("Cannot locate field " + fieldName + " in subDomain " + name);
        }
    }

    inline const Field& GetField(PetscInt id, FieldType type = FieldType::SOL) const {
        auto field = std::find_if(fieldsByName.begin(), fieldsByName.end(), [type, id](auto pair) { return pair.second.type == type && pair.second.id == id; });
        if (field != fieldsByName.end()) {
            return field->second;
        } else {
            throw std::invalid_argument("Cannot locate field with id " + std::to_string(id) + " in subDomain " + name);
        }
    }

    inline const std::vector<Field>& GetFields(FieldType type = FieldType::SOL) const { return fieldsByType.at(type); }

    /**
     * Get the petscField object from the dm or auxDm for this region
     * @param fieldName
     * @return
     */
    PetscObject GetPetscFieldObject(const Field& field);

    inline const Field& GetSolutionField(const std::string& fieldName) const {
        if (auto domainPtr = domain.lock()) {
            return domainPtr->GetSolutionField(fieldName);
        } else {
            throw std::runtime_error("Cannot GetSolutionField. Domain is expired.");
        }
    }

    //[[deprecated("Should remove need for direct dm access")]]
    DM& GetDM();
    //[[deprecated("Should remove need for direct dm access")]]
    DM GetAuxDM();
    Vec GetSolutionVector();
    Vec GetAuxVector();

    // Function to create/get the subDM.  If there is no subDM the dm will be returned
    DM GetSubDM();
    Vec GetSubSolutionVector();

    // returns an aux vector with the correct data sized for the subDM
    DM GetSubAuxDM();
    Vec GetSubAuxVector();

    // Get the discreteSystem describe system for this subDomain
    inline PetscDS GetDiscreteSystem() { return discreteSystem; }

    inline DMLabel GetLabel() { return label; }

    /**
     * Get an aux DS if it is available
     * @return
     */
    inline PetscDS GetAuxDiscreteSystem() { return auxDiscreteSystem; }

    PetscInt GetDimensions() const;
    inline PetscInt GetNumberFields() const { return fieldsByName.size(); }

    inline MPI_Comm GetComm() { return PetscObjectComm((PetscObject)GetDM()); }

    /**
     * Support function to project the fields on to the global vector
     */
    void ProjectFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, Vec globVec, PetscReal time = 0.0);
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_SUBDOMAIN_HPP
