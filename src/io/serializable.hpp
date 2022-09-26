#ifndef ABLATELIBRARY_SERIALIZABLE_HPP
#define ABLATELIBRARY_SERIALIZABLE_HPP

#include <petsc.h>
#include <string>

namespace ablate::io {
/**
 * This class gives the option to serialize/save restore.  A bool is used at startup to determine if it should be save/restored
 */
class Serializable {
   public:
    virtual ~Serializable() = default;
    /**
     * boolean used to determined if this object should be serialized at runtime
     * @return
     */
    [[nodiscard]] virtual bool Serialize() const { return true; }

    /**
     * only required function, returns the id of the object.  Should be unique for the simulation
     * @return
     */
    [[nodiscard]] virtual const std::string& GetId() const = 0;

    /**
     * Save the state to the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    virtual void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) = 0;

    /**
     * Restore the state from the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    virtual void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) = 0;
};
}  // namespace ablate::io

#endif  // ABLATELIBRARY_SERIALIZABLE_HPP
