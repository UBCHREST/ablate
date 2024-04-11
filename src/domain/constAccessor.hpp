#ifndef ABLATELIBRARY_CONSTACCESSOR_HPP
#define ABLATELIBRARY_CONSTACCESSOR_HPP

#include <petsc.h>
#include "field.hpp"

namespace ablate::domain {
/**
 * Class responsible for computing point data locations for dm plex integration
 */
template <class DataType, bool isLocal = true>
class ConstAccessor {
   private:
    /**
     * The petsc vector used to hold the data
     */
    Vec dataVector;

    /**
     * The extracted data from the vector
     */
    const PetscScalar* dataArray;

    /**
     * Store the DM for the vector to find memory locations
     */
    DM dataDM;

   public:
    explicit ConstAccessor(Vec dataVectorIn, DM dm = nullptr) : dataVector(dataVectorIn), dataDM(dm) {
        // Get read/write access to the vector
        VecGetArrayRead(dataVector, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

        // Get the dm from the vector if not specified
        if (!dataDM) {
            VecGetDM(dataVector, &dataDM) >> ablate::utilities::PetscUtilities::checkError;
        }
    }

    ~ConstAccessor() {
        // Put back the vector
        VecRestoreArrayRead(dataVector, &dataArray) >> ablate::utilities::PetscUtilities::checkError;
    };

    //! Inline function to compute the memory address at this point
    template <class IndexType>
    inline const DataType* operator[](IndexType point) const {
        const DataType* field;
        if constexpr (isLocal) {
            DMPlexPointLocalRead(dataDM, point, dataArray, &field) >> ablate::utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointGlobalRead(dataDM, point, dataArray, &field) >> ablate::utilities::PetscUtilities::checkError;
        }
        return field;
    }

    //! Inline function to compute the memory address at this point using PETSC style return error
    template <class IndexType>
    inline PetscErrorCode operator()(IndexType point, const DataType** field) const {
        PetscFunctionBeginHot;
        if constexpr (isLocal) {
            PetscCall(DMPlexPointLocalRead(dataDM, point, dataArray, field));
        } else {
            PetscCall(DMPlexPointGlobalRead(dataDM, point, dataArray, field));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    /**
     * prevent copy of this class
     */
    ConstAccessor(const ConstAccessor&) = delete;
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELDACCESSOR_HPP
