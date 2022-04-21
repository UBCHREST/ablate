#ifndef ABLATELIBRARY_LOGGABLE_HPP
#define ABLATELIBRARY_LOGGABLE_HPP
#include <petsc.h>
#include "demangler.hpp"
#include "petscError.hpp"

namespace ablate::utilities {
template <class T>
class Loggable {
   private:
    inline static PetscClassId petscClassId = 0;

    // Store a single depth active event
    PetscLogEvent activeEvent = PETSC_DECIDE;

   protected:
    Loggable() {
        if (petscClassId == 0) {
            auto className = utilities::Demangler::Demangle(typeid(T).name());
            PetscClassIdRegister(className.c_str(), &petscClassId) >> checkError;
        }
    }

    inline const PetscClassId& GetPetscClassId() const { return petscClassId; }

    inline PetscLogEvent RegisterEvent(const char* eventName) {
        PetscLogEvent eventId;
        PetscLogEventRegister(eventName, petscClassId, &eventId) >> checkError;
        return eventId;
    }

    inline void StartEvent(const char* eventName) {
        if (activeEvent == PETSC_DECIDE) {
            PetscLogEventRegister(eventName, petscClassId, &activeEvent) >> checkError;
            PetscLogEventBegin(activeEvent, 0, 0, 0, 0) >> checkError;
        } else {
            throw std::runtime_error("Cannot Start Event, an event is already active.");
        }
    }

    inline void EndEvent() {
        if (activeEvent > 0) {
            PetscLogEventEnd(activeEvent, 0, 0, 0, 0);
            activeEvent = PETSC_DECIDE;
        } else {
            throw std::runtime_error("Cannot End Event.  No active event.");
        }
    }
};
};  // namespace ablate::utilities

#endif  // ABLATELIBRARY_LOGGABLE_HPP
