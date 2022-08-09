#include "kokkosUtilities.hpp"
#include <Kokkos_Core.hpp>
#include "environment/runEnvironment.hpp"

void ablate::utilities::KokkosUtilities::Initialize() {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize(*ablate::environment::RunEnvironment::GetArgCount(), *ablate::environment::RunEnvironment::GetArgs());

        ablate::environment::RunEnvironment::RegisterCleanUpFunction("ablate::utilities::KokkosUtilities::Initialize", []() { Kokkos::finalize(); });
    }
}
