#ifndef ABLATELIBRARY_FIELDACCESSOR_HPP
#define ABLATELIBRARY_FIELDACCESSOR_HPP

#include <petsc.h>
#include "field.hpp"

namespace ablate::domain {
/**
 * Class responsible for computing point data locations for dm plex integration
 */
template <class DataType, bool isLocal = true>
class FieldAccessor {
   private:
    /**
     * The petsc vector used to hold the data
     */
    Vec dataVector;

    /**
     * The field information needed to extract the data
     */
    const Field& dataField;

    /**
     * The extracted data from the vector
     */
    PetscScalar* dataArray{};

    /**
     * Store the DM for the vector to find memory locations
     */
    DM dataDM;

   public:
    FieldAccessor(Vec dataVectorIn, const Field& dataFieldIn, DM dm = nullptr) : dataVector(dataVectorIn), dataField(dataFieldIn), dataDM(dm) {
        // Get read/write access to the vector
        VecGetArray(dataVector, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

        // Get the dm from the vector if not specified
        if (!dataDM) {
            VecGetDM(dataVector, &dataDM) >> ablate::utilities::PetscUtilities::checkError;
        }
    }

    ~FieldAccessor() {
        // Put back the vector
        VecRestoreArray(dataVector, &dataArray) >> ablate::utilities::PetscUtilities::checkError;
    };

    //! provide easy access to the field information
    [[nodiscard]] inline const Field& GetField() const { return dataField; }

    //! Inline function to compute the memory address at this point
    template <class IndexType>
    inline DataType* operator[](IndexType point) {
        DataType* field;
        if constexpr (isLocal) {
            DMPlexPointLocalFieldRef(dataDM, point, dataField.id, dataArray, &field) >> ablate::utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointGlobalFieldRef(dataDM, point, dataField.id, dataArray, &field) >> ablate::utilities::PetscUtilities::checkError;
        }
        return field;
    }

    //! Inline function to compute the memory address at this point using PETSC style return error
    template <class IndexType>
    inline PetscErrorCode operator()(IndexType point, DataType** field) {
        PetscFunctionBeginHot;
        if constexpr (isLocal) {
            PetscCall(DMPlexPointLocalFieldRef(dataDM, point, dataField.id, dataArray, field));
        } else {
            PetscCall(DMPlexPointGlobalFieldRef(dataDM, point, dataField.id, dataArray, field));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    /**
     * prevent copy of this class
     */
    FieldAccessor(const FieldAccessor&) = delete;
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELDACCESSOR_HPP
