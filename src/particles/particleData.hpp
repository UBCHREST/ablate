#ifndef ABLATELIBRARY_PARTICLEDATA_HPP
#define ABLATELIBRARY_PARTICLEDATA_HPP

#include <petsc.h>
#include <map>
#include "particles/field.hpp"
#include "utilities/petscError.hpp"

namespace ablate::particles {
/**
 * Computes the memory location for particle field data
 */
struct ParticleData {
    //! Store a pointer to the field
    const Field& field;

    //! optional reference to dwSwarm.  If this is provided the particle data will cleanup after itself
    const DM swarmDm;

    //! the array for the solution values
    PetscScalar* values;

    /**
     * Get the particle data from the field
     * @param field
     * @param swarmDm
     */
    ParticleData(const Field& field, DM swarmDm) : field(field), swarmDm(swarmDm), values(nullptr) { DMSwarmGetField(swarmDm, field.name.c_str(), nullptr, nullptr, (void**)&values) >> checkError; }

    ParticleData(const Field& field, PetscScalar* values) : field(field), swarmDm(nullptr), values(values) {}

    ~ParticleData() {
        if (swarmDm) {
            DMSwarmRestoreField(swarmDm, field.name.c_str(), nullptr, nullptr, (void**)&values) >> checkError;
        }
    }

    //! Inline function to compute the memory address at this particle
    template <class IndexType>
    inline PetscScalar* operator[](IndexType particle) const {
        return values + field[particle];
    }

    /**
     * Return the value at this particle
     * @tparam IndexType
     * @param particle
     * @return
     */
    template <class IndexType>
    inline PetscScalar& operator()(IndexType particle) {
        return values[field[particle]];
    }

    /**
     * Return the value at this particle
     * @tparam IndexType
     * @param particle
     * @return
     */
    template <class IndexType, class DimType>
    inline PetscScalar& operator()(IndexType particle, DimType dim) {
        return values[field[particle] + dim];
    }

    /**
     * Return the value at this particle
     * @tparam IndexType
     * @param particle
     * @return
     */
    template <class IndexType>
    inline const PetscScalar& operator()(IndexType particle) const {
        return values[field[particle]];
    }

    /**
     * Return the value at this particle
     * @tparam IndexType
     * @param particle
     * @return
     */
    template <class IndexType, class DimType>
    inline const PetscScalar& operator()(IndexType particle, DimType dim) const {
        return values[field[particle] + dim];
    }

    /**
     * prevent copy of this class
     */
    ParticleData(const ParticleData&) = delete;
};
}  // namespace ablate::particles::processes
#endif  // ABLATELIBRARY_SWARMDATA_HPP
