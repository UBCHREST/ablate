#ifndef ABLATELIBRARY_SUBDOMAIN_HPP
#define ABLATELIBRARY_SUBDOMAIN_HPP
#include <petsc.h>
#include <algorithm>
#include <map>
#include <mathFunctions/fieldFunction.hpp>
#include <memory>
#include <string>
#include <utilities/petscError.hpp>
#include "domain.hpp"
#include "fieldDescription.hpp"

namespace ablate::domain {

class SubDomain {
   private:
    // const reference to the parent domain/dm
    Domain& domain;

    // Keep a name for this subDomain for output/debug
    std::string name;

    // The label used to describe this subDomain
    DMLabel label;
    PetscInt labelValue;

    // contains the DM field numbers for the fields in this DS, or NULL
    IS fieldMap;

    // Keep track of fields that live in this subDomain;
    std::map<std::string, Field> fieldsByName;

    // also keep the fields by type for faster iteration
    std::map<FieldLocation, std::vector<Field>> fieldsByType;

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
    void CopyGlobalToSubVector(DM subDM, DM gDM, Vec subVec, Vec globVec, const std::vector<Field>& subFields, const std::vector<Field>& gFields = {}, bool localVector = false) const;

   public:
    SubDomain(Domain& domain, PetscInt dsNumber, std::vector<std::shared_ptr<FieldDescription>> allAuxFields);
    ~SubDomain();

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

    inline const Field& GetField(PetscInt id, FieldLocation location = FieldLocation::SOL) const {
        auto field = std::find_if(fieldsByName.begin(), fieldsByName.end(), [location, id](auto pair) { return pair.second.location == location && pair.second.id == id; });
        if (field != fieldsByName.end()) {
            return field->second;
        } else {
            throw std::invalid_argument("Cannot locate field with id " + std::to_string(id) + " in subDomain " + name);
        }
    }

    inline const std::vector<Field>& GetFields(FieldLocation type = FieldLocation::SOL) const { return fieldsByType.at(type); }

    // Helper function that returns the dm or auxDM
    inline DM GetFieldDM(const Field& field) noexcept {
        switch (field.location) {
            case FieldLocation::SOL:
                return GetDM();
            case FieldLocation::AUX:
                return GetAuxDM();
        }
        return nullptr;
    }

    // Helper function that returns the vec or auxVec
    inline Vec GetFieldVec(const Field& field) noexcept {
        switch (field.location) {
            case FieldLocation::SOL:
                return GetSolutionVector();
            case FieldLocation::AUX:
                return GetAuxVector();
        }
        return nullptr;
    }

    // return true if the field was defined
    inline bool ContainsField(const std::string& fieldName) { return fieldsByName.count(fieldName) > 0; }

    /**
     * Helper function that checks to see if any part of the specified region is in this subDomain
     */
    bool InRegion(const domain::Region&) const;

    /**
     * Get the petscField object from the dm or auxDm for this region
     * @param fieldName
     * @return
     */
    PetscObject GetPetscFieldObject(const Field& field);

    inline DM& GetDM() noexcept { return domain.GetDM(); }
    inline DM GetAuxDM() noexcept { return auxDM; }
    Vec GetSolutionVector() noexcept;
    Vec GetAuxVector() noexcept;

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
     * determines if this point is in this region as defined by the label and labelID
     * @param point
     * @return
     */
    inline bool InRegion(PetscInt point) {
        if (!label) {
            return true;
        }
        PetscInt ptValue;
        DMLabelGetValue(label, point, &ptValue) >> checkError;
        return ptValue == labelValue;
    }

    /**
     * Support function to project the fields on to the global vector
     */
    void ProjectFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, Vec globVec, PetscReal time = 0.0, const std::shared_ptr<domain::Region> region = {});

    /**
     * Support function to project the fields on to vector that lives only on the subDM
     */
    void ProjectFieldFunctionsToSubDM(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, Vec globVec, PetscReal time = 0.0);
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_SUBDOMAIN_HPP
