#ifndef ABLATELIBRARY_CONSTFIELDACCESSOR_HPP
#define ABLATELIBRARY_CONSTFIELDACCESSOR_HPP

#include <petsc.h>
#include "field.hpp"

namespace ablate::domain {
/**
 * Class responsible for computing const point data locations for dm plex integration
 */
template <class DataType, bool isLocal = true>
class ConstFieldAccessor {
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
    const PetscScalar* dataArray{};

    /**
     * Store the DM for the vector to find memory locations
     */
    DM dataDM;

   public:
    ConstFieldAccessor(Vec dataVectorIn, const Field& dataFieldIn, DM dm = nullptr) : dataVector(dataVectorIn), dataField(dataFieldIn), dataDM(dm) {
        // Get read/write access to the vector
        VecGetArrayRead(dataVector, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

        // Get the dm from the vector if not specified
        if (!dataDM) {
            VecGetDM(dataVector, &dataDM) >> ablate::utilities::PetscUtilities::checkError;
        }
    }

    ~ConstFieldAccessor() {
        // Put back the vector
        VecRestoreArrayRead(dataVector, &dataArray) >> ablate::utilities::PetscUtilities::checkError;
    };

    //! provide easy access to the field information
    [[nodiscard]] inline const Field& GetField() const { return dataField; }

    //! Inline function to compute the memory address at this point
    template <class IndexType>
    inline const DataType* operator[](IndexType point) const {
        const DataType* field;
        if constexpr (isLocal) {
            DMPlexPointLocalFieldRead(dataDM, point, dataField.id, dataArray, &field) >> ablate::utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointGlobalFieldRead(dataDM, point, dataField.id, dataArray, &field) >> ablate::utilities::PetscUtilities::checkError;
        }
        return field;
    }

    //! Inline function to compute the memory address at this point using PETSC style return error
    template <class IndexType>
    inline PetscErrorCode operator()(IndexType point, const DataType** field) const {
        PetscFunctionBeginHot;
        if constexpr (isLocal) {
            PetscCall(DMPlexPointLocalFieldRead(dataDM, point, dataField.id, dataArray, field));
        } else {
            PetscCall(DMPlexPointGlobalFieldRead(dataDM, point, dataField.id, dataArray, field));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    /**
     * prevent copy of this class
     */
    ConstFieldAccessor(const ConstFieldAccessor&) = delete;
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_CONSTFIELDACCESSOR_HPP
