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
#include "io/serializable.hpp"

namespace ablate::domain {

class SubDomain : public io::Serializable {
   private:
    //! store the default name of the domain
    inline static const std::string defaultName = "domain";

    //! const reference to the parent domain/dm
    Domain& domain;

    //! Keep a name for this subDomain for output/debug
    std::string name;

    //! The label used to describe this subDomain
    DMLabel label;

    //! label value used describe this subDomain
    PetscInt labelValue;

    //! contains the DM field numbers for the fields in this DS, or NULL
    IS fieldMap;

    //! Keep track of fields that live in this subDomain;
    std::map<std::string, Field> fieldsByName;

    //! also keep the fields by type for faster iteration
    std::map<FieldLocation, std::vector<Field>> fieldsByType;

    //! Each subDomain will operate over a ds
    PetscDS discreteSystem;
    PetscDS auxDiscreteSystem{};

    //! The auxDm and auxVec are for only this subDomain
    DM auxDM;

    //! The local auxVector defined only over this subdomain.  This is the canonical source of information
    Vec auxLocalVec;

    //! The global auxVector defined only over this subdomain
    Vec auxGlobalVec;

    //! Create/store a subDM for output
    DM subDM;
    Vec subSolutionVec;
    DM subAuxDM;
    Vec subAuxVec;

    //! store any exact solutions for io
    std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions;

    /**
     * support call to copy from global to sub vec
     * @param subDM
     * @param gDM
     * @param subVec
     * @param globVec
     * @param subFields
     * @param gFields
     * @param localVector
     */
    void CopyGlobalToSubVector(DM subDM, DM gDM, Vec subVec, Vec globVec, const std::vector<Field>& subFields, const std::vector<Field>& gFields = {}, bool localVector = false) const;

    /**
     * support call to copy from sub vec to global
     * @param subDM
     * @param gDM
     * @param subVec
     * @param globVec
     * @param subFields
     * @param gFields
     * @param localVector
     */
    void CopySubVectorToGlobal(DM subDM, DM gDM, Vec subVec, Vec globVec, const std::vector<Field>& subFields, const std::vector<Field>& gFields = {}, bool localVector = false) const;

   public:
    SubDomain(Domain& domain, PetscInt dsNumber, const std::vector<std::shared_ptr<FieldDescription>>& allAuxFields);
    ~SubDomain() override;

    /**
     * Create the auxDM, auxVec, and other structures on the subDomain
     */
    void CreateSubDomainStructures();

    /**
     * returns a references to the field (sol/aux) for a given field name
     * @param fieldName the string name of the field
     * @return a reference to the Field object
     */
    [[nodiscard]] inline const Field& GetField(const std::string& fieldName) const {
        if (fieldsByName.count(fieldName)) {
            return fieldsByName.at(fieldName);
        } else {
            throw std::invalid_argument("Cannot locate field " + fieldName + " in subDomain " + name);
        }
    }

    /**
     * returns a references to the field for a given id and location
     * @param id
     * @param location
     * @return
     */
    [[nodiscard]] inline const Field& GetField(PetscInt id, FieldLocation location = FieldLocation::SOL) const {
        auto field = std::find_if(fieldsByName.begin(), fieldsByName.end(), [location, id](auto pair) { return pair.second.location == location && pair.second.id == id; });
        if (field != fieldsByName.end()) {
            return field->second;
        } else {
            throw std::invalid_argument("Cannot locate field with id " + std::to_string(id) + " in subDomain " + name);
        }
    }

    /**
     * returns all fields of a certain type
     * @param type (defaults to SOL)
     * @return
     */
    [[nodiscard]] inline const std::vector<Field>& GetFields(FieldLocation type = FieldLocation::SOL) const { return fieldsByType.at(type); }

    /**
     * returns all fields of a certain type with a specific tag.  This can be costly and should be used only for setup
     * @param type
     * @param tag
     * @return
     */
    [[nodiscard]] std::vector<Field> GetFields(FieldLocation type, std::string_view tag) const;

    /**
     * gets the dm corresponding to a field location (aux/sol)
     * @param field
     * @return
     */
    inline DM GetFieldDM(const Field& field) noexcept {
        switch (field.location) {
            case FieldLocation::SOL:
                return GetDM();
            case FieldLocation::AUX:
                return GetAuxDM();
        }
        return nullptr;
    }

    /**
     * Returns the global solution field or the local aux vector depending upon the field
     * @param field
     * @return
     */
    inline Vec GetVec(const Field& field) noexcept {
        switch (field.location) {
            case FieldLocation::SOL:
                return GetSolutionVector();
            case FieldLocation::AUX:
                return GetAuxVector();
        }
        return nullptr;
    }

    /**
     * Returns always returns a global vector of sol or aux
     * @param field
     * @return
     */
    inline Vec GetGlobalVec(const Field& field) noexcept {
        switch (field.location) {
            case FieldLocation::SOL:
                return GetSolutionVector();
            case FieldLocation::AUX:
                return GetAuxGlobalVector();
        }
        return nullptr;
    }

    /**
     * Determine if the field is available as either a sol or aux field
     * @param fieldName
     * @return
     */
    inline bool ContainsField(const std::string& fieldName) { return fieldsByName.count(fieldName) > 0; }

    /**
     * Determine if the provided region lives inside of subdomain region
     * @return
     */
    bool InRegion(const domain::Region&) const;

    /**
     * determines if this point is in this region as defined by the label and labelID
     * @param point
     * @return
     */
    inline bool InRegion(PetscInt point) const {
        if (!label) {
            return true;
        }
        PetscInt ptValue;
        DMLabelGetValue(label, point, &ptValue) >> checkError;
        return ptValue == labelValue;
    }

    /**
     * The comm used to define this subDomain and resulting solvers
     * @return
     */
    inline MPI_Comm GetComm() const { return PetscObjectComm((PetscObject)domain.GetDM()); }

    /**
     * The label (if any) used to define this subDomain
     * @return
     */
    inline DMLabel GetLabel() { return label; }

    /**
     * Return the id describing this region
     * @return
     */
    inline const std::string& GetId() const override { return name; }

    /**
     * Returns the number of physical dimensions defining the dm
     * @return
     */
    inline PetscInt GetDimensions() const noexcept { return domain.GetDimensions(); }

    /**
     * Get the petscField object from the dm or auxDm for this region
     * @param fieldName
     * @return
     */
    PetscObject GetPetscFieldObject(const Field& field);

    /**
     * Returns raw access to the global dm
     * @return
     */
    inline DM& GetDM() const noexcept { return domain.GetDM(); }

    /**
     * Returns the dm describing the aux fields living in this subdomain.  The dm is defined across
     * the entire mesh, but the fields are only define under this subdomain
     * @return
     */
    inline DM GetAuxDM() noexcept { return auxDM; }

    /**
     * Returns the global solution vector
     * @return
     */
    inline Vec GetSolutionVector() noexcept { return domain.GetSolutionVector(); }

    /**
     * Returns the local aux vector
     * @return
     */
    inline Vec GetAuxVector() noexcept { return auxLocalVec; }

    /**
     * Return the global aux vector with information updated from the auxLocVec
     */
    Vec GetAuxGlobalVector();

    /**
     * The SubDM is defined as a dm that lives only over this subdomain region.  It may be useful for outputting. This function will create (if needed) and
     * return the subdomain. If a subdomain is not needed (same ds over entire dm) the global dm is returned.
     * @return
     */
    DM GetSubDM();

    /**
     * Returns a "global" solution vector defined over the subDomain
     * @return
     */
    Vec GetSubSolutionVector();

    /**
     * The SubDM is defined as a dm that lives only over this subdomain region.  It may be useful for outputting. This function will create (if needed) and
     * return the subdomain for the aux variables. If a subdomain is not needed (same ds over entire dm) the global aux dm is returned.
     * @return
     */
    DM GetSubAuxDM();

    /**
     * Returns a "global" aux vector defined over the subDomain
     * @return
     */
    Vec GetSubAuxVector();

    /**
     * Get the discreteSystem describe system for this subDomain
     * @return
     */
    inline PetscDS GetDiscreteSystem() { return discreteSystem; }

    /**
     * Get an aux DS if it is available
     * @return
     */
    inline PetscDS GetAuxDiscreteSystem() { return auxDiscreteSystem; }

    /**
     * The number of aux and solution fields in this subdomain
     * @return
     */
    inline PetscInt GetNumberFields() const { return fieldsByName.size(); }

    /**
     * project the list of field function into the provided local vector.  Allows solution and aux vectors
     * @param fieldFunctions
     * @param globVec
     */
    void ProjectFieldFunctionsToLocalVector(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, Vec locVec, PetscReal time = 0.0) const;

    /**
     * Support function to project the fields on to vector that lives only on the subDM
     * @param initialization
     * @param globVec
     * @param time
     */
    void ProjectFieldFunctionsToSubDM(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, Vec globVec, PetscReal time = 0.0);

    /**
     * set exactSolutions if the fields live in the ds
     * @param exactSolutions
     */
    void SetsExactSolutions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& exactSolutions);

    /**
     * Get a global vector with only a single field
     * @param vecIs
     * @param vec
     * @param subdm
     * @return
     */
    PetscErrorCode GetFieldGlobalVector(const Field&, IS* vecIs, Vec* vec, DM* subdm);

    /**
     * Restore a global vector with only a single field
     * @param vecIs
     * @param vec
     * @param subdm
     * @return
     */
    PetscErrorCode RestoreFieldGlobalVector(const Field&, IS* vecIs, Vec* vec, DM* subdm);

    /**
     * Get a local vector (with boundary values)  with only a single field
     * @param vecIs
     * @param time time is ued to insert boundary conditions for the global solution vector
     * @param vec
     * @param subdm
     * @return
     */
    PetscErrorCode GetFieldLocalVector(const Field&, PetscReal time, IS* vecIs, Vec* vec, DM* subdm);

    /**
     * Restore a local vector with only a single field
     * @param vecIs
     * @param vec
     * @param subdm
     * @return
     */
    PetscErrorCode RestoreFieldLocalVector(const Field&, IS* vecIs, Vec* vec, DM* subdm);

    /**
     * Serialization save
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * Serialization restore
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * This checks for whether the label describing the subdomain exists. If it does, use DMPlexFilter. If not, use DMClone to return new DM.
     * @param inDM
     */
    void CreateEmptySubDM(DM* inDM, std::shared_ptr<domain::Region> region = {});
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_SUBDOMAIN_HPP
