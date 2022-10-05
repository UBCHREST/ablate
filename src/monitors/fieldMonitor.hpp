#ifndef ABLATELIBRARY_FIELDMONITOR_HPP
#define ABLATELIBRARY_FIELDMONITOR_HPP

#include <petsc.h>
#include <vector>
#include "io/serializable.hpp"
#include "monitor.hpp"

namespace ablate::monitors {

/**
 * An abstract class that provides support for creating, writing, and outputting new fields
 */
class FieldMonitor : public Monitor, public io::Serializable {
   private:
    // Reuse the domain object to set up a domain to hold the monitor vector/dm
    std::shared_ptr<ablate::domain::Domain> monitorDomain = nullptr;

   protected:
    // Hold onto the subdomain for this monitor.  It should be over the entire monitorDomain
    std::shared_ptr<ablate::domain::SubDomain> monitorSubDomain = nullptr;

   public:
    /**
     * only required function, returns the id of the object.  Should be unique for the simulation
     * @return
     */
    const std::string& GetId() const override { return monitorDomain->GetName(); }

    /**
     * In order to use the base class, the Register call must be overridden and Register(std::shared_ptr<solver::Solver> solverIn, std::vector<domain::FieldDescription> fields) must be called from the
     * method
     * @param solverIn
     */
    void Register(std::shared_ptr<solver::Solver> solverIn) override = 0;

    /**
     * In order to use the base class, the Register call must be overridden and
     * @param solverIn
     */
    void Register(std::string id, std::shared_ptr<solver::Solver> solverIn, std::vector<std::shared_ptr<domain::FieldDescriptor>> fields);

    /**
     * Save the state to the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    virtual void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * Restore the state from the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    virtual void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_FIELDMONITOR_HPP
