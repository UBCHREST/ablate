#ifndef ABLATELIBRARY_ACCESSOR_HPP
#define ABLATELIBRARY_ACCESSOR_HPP

#include <petsc.h>
#include "field.hpp"

namespace ablate::domain {
/**
 * Class responsible for computing point data locations for dm plex integration
 */
template <class DataType, bool isLocal = true>
class Accessor {
   private:
    /**
     * The petsc vector used to hold the data
     */
    Vec dataVector;

    /**
     * The extracted data from the vector
     */
    PetscScalar* dataArray;

    /**
     * Store the DM for the vector to find memory locations
     */
    DM dataDM;

   public:
    Accessor(Vec dataVectorIn, DM dm = nullptr) : dataVector(dataVectorIn), dataDM(dm) {
        // Get read/write access to the vector
        VecGetArray(dataVector, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

        // Get the dm from the vector if not specified
        if (!dataDM) {
            VecGetDM(dataVector, &dataDM) >> ablate::utilities::PetscUtilities::checkError;
        }
    }

    ~Accessor() {
        // Put back the vector
        VecRestoreArray(dataVector, &dataArray) >> ablate::utilities::PetscUtilities::checkError;
    };

    //! Inline function to compute the memory address at this point
    template <class IndexType>
    inline DataType* operator[](IndexType point) {
        DataType* field;
        if constexpr (isLocal) {
            DMPlexPointLocalRef(dataDM, point, dataArray, &field) >> ablate::utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointGlobalRef(dataDM, point, dataArray, &field) >> ablate::utilities::PetscUtilities::checkError;
        }
        return field;
    }

    //! Inline function to compute the memory address at this point using PETSC style return error
    template <class IndexType>
    inline PetscErrorCode operator()(IndexType point, DataType** field) {
        PetscFunctionBeginHot;
        if constexpr (isLocal) {
            PetscCall(DMPlexPointLocalRef(dataDM, point, dataArray, field));
        } else {
            PetscCall(DMPlexPointGlobalRef(dataDM, point, dataArray, field));
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    /**
     * prevent copy of this class
     */
    Accessor(const Accessor&) = delete;
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELDACCESSOR_HPP
