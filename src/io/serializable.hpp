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
    virtual PetscErrorCode Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) = 0;

    /**
     * Restore the state from the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    virtual PetscErrorCode Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) = 0;

   protected:
    /**
     * helper function to save PetscScalar to a PetscViewer. It is assumed to be the same value across all mpi ranks
     * @param viewer
     * @param name
     * @param value
     */
    static PetscErrorCode SaveKeyValue(PetscViewer viewer, const char* name, PetscScalar value);

    /**
     * helper function to restore PetscScalar to a PetscViewer. The same value is returned on all mpi ranks
     * @param viewer
     * @param name
     * @param value
     */
    static PetscErrorCode RestoreKeyValue(PetscViewer viewer, const char* name, PetscScalar& value);

    /**
     * Helper function to save a single key/value pair to the PetscViewer.  It is assumed to be the same value across all mpi ranks
     * @tparam T
     * @param viewer
     * @param name
     * @param value
     */
    template <class T>
    static inline PetscErrorCode SaveKeyValue(PetscViewer viewer, const char* name, T value) {
        PetscFunctionBeginUser;
        auto tempValue = (PetscScalar)value;
        PetscCall(SaveKeyValue(viewer, name, tempValue));
        PetscFunctionReturn(0);
    }

    /**
     * Helper function to save a restore key/value pair to the PetscViewer.  It is assumed to be the same value across all mpi ranks
     * @tparam T
     * @param viewer
     * @param name
     * @param value
     */
    template <class T>
    static inline PetscErrorCode RestoreKeyValue(PetscViewer viewer, const char* name, T& value) {
        PetscFunctionBeginUser;
        PetscScalar tempValue = {};
        PetscCall(RestoreKeyValue(viewer, name, tempValue));
        value = (T)tempValue;
        PetscFunctionReturn(0);
    }
};
}  // namespace ablate::io

#endif  // ABLATELIBRARY_SERIALIZABLE_HPP
