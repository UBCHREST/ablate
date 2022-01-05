#ifndef ABLATELIBRARY_SERIALIZABLE_HPP
#define ABLATELIBRARY_SERIALIZABLE_HPP

namespace ablate::io {
class Serializable {
   public:
    virtual ~Serializable() = default;
    virtual const std::string& GetId() const = 0;
    virtual void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) = 0;
    virtual void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) = 0;
};
}  // namespace ablate::io

#endif  // ABLATELIBRARY_SERIALIZABLE_HPP
