#ifndef ABLATELIBRARY_POINTDATA_HPP
#define ABLATELIBRARY_POINTDATA_HPP

#include <petsc.h>
#include <map>
#include "particles/field.hpp"
#include "utilities/petscUtilities.hpp"

namespace ablate::particles::accessors {
/**
 * Computes the memory location for particle field data
 */
template <class DataType>
struct Data {
    //! the array for the solution values
    DataType* values = nullptr;

    //! The number of the components
    PetscInt numberComponents = 0;

    //! The size of the component for this data
    PetscInt dataSize = 0;

    //! The offset in the local array, 0 for aux, computed for sol
    PetscInt offset = 0;

    /**
     * empty default constructor
     */
    Data() {}

    /**
     * The default constructor
     * @param values
     * @param numberComponents
     * @param dataSizeIn
     * @param offset
     */
    Data(DataType* values, PetscInt numberComponents, PetscInt dataSizeIn = 0, PetscInt offset = 0)
        : values(values), numberComponents(numberComponents), dataSize(dataSizeIn ? dataSizeIn : numberComponents), offset(offset) {}

    /**
     * Takes the input values from the particle field
     * @param values
     * @param numberComponents
     * @param dataSizeIn
     * @param offset
     */
    Data(DataType* values, const ablate::particles::Field& field) : values(values), numberComponents(field.numberComponents), dataSize(field.dataSize), offset(field.offset) {}

    //! Inline function to compute the memory address at this particle
    template <class IndexType>
    inline DataType* operator[](IndexType particle) const {
        return values + (particle * dataSize + offset);
    }

    /**
     * Return the value at this particle
     * @tparam IndexType
     * @param particle
     * @return
     */
    template <class IndexType>
    inline DataType& operator()(IndexType particle) {
        return values[particle * dataSize + offset];
    }

    /**
     * Return the value at this particle
     * @tparam IndexType
     * @param particle
     * @return
     */
    template <class IndexType, class DimType>
    inline DataType& operator()(IndexType particle, DimType dim) {
        return values[particle * dataSize + offset + dim];
    }

    /**
     * Copy all of the dimensions of this field to specified destination
     * @tparam IndexType
     * @param source
     * @param np the number of particles to copy
     */
    template <class DestinationDataType, class IndexType>
    inline void CopyFrom(DestinationDataType* source, IndexType p) const {
        for (PetscInt d = 0; d < numberComponents; d++) {
            values[p * dataSize + offset + d] = source[d];
        }
    }

    /**
     * Adds all of the dimensions of this field to specified destination
     * @tparam IndexType
     * @param source
     * @param np the number of particles to copy
     */
    template <class DestinationDataType, class IndexType>
    inline void AddFrom(DestinationDataType* source, IndexType p) const {
        for (PetscInt d = 0; d < numberComponents; d++) {
            values[p * dataSize + offset + d] += source[d];
        }
    }

    /**
     * Copy the values in this pointData to destination
     * @tparam IndexType
     * @param destination
     * @param np the number of particles to copy
     */
    template <class DestinationDataType, class IndexType>
    inline void CopyAll(DestinationDataType* destination, IndexType np) const {
        for (IndexType p = 0; p < np; p++) {
            for (PetscInt d = 0; d < numberComponents; d++) {
                destination[p * numberComponents + d] = values[p * dataSize + offset + d];
            }
        }
    }
};

/**
 * Predefine the two most common types of data
 */
using ConstPointData = Data<const PetscReal>;
using PointData = Data<PetscReal>;

}  // namespace ablate::particles::accessors
#endif  // ABLATELIBRARY_SWARMDATA_HPP
