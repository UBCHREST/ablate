#ifndef ABLATELIBRARY_TCHEM2_SPEEDOFSOUND_HPP
#define ABLATELIBRARY_TCHEM2_SPEEDOFSOUND_HPP

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace ablate::eos::tChem2 {

struct SpeedOfSound {
    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
    using device_type = typename Tines::UseThisDevice<exec_space>::type;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    using real_type_1d_view_host_type = Tines::value_type_1d_view<real_type, host_device_type>;
    using real_type_2d_view_host_type = Tines::value_type_2d_view<real_type, host_device_type>;

    using kinetic_model_type = KineticModelConstData<device_type>;

    using kinetic_model_host_type = KineticModelConstData<host_device_type>;

    static inline ordinal_type getWorkSpaceSize(ordinal_type numberSpecies) { return numberSpecies; }

    /**
     * tchem like function to compute sensible internal energy on device
     * @param policy
     * @param state
     * @param internalEnergyRef
     * @param enthalpyMass
     * @param temperature
     * @param kmcd
     */
    [[maybe_unused]] static void runDeviceBatch(  /// thread block size
        typename UseThisTeamPolicy<exec_space>::type& policy,
        //// input
        const real_type_2d_view_type& state,
        /// output
        const real_type_1d_view_type& speedOfSound,
        /// const data from kinetic model
        const kinetic_model_type& kmcd);

    /**
     * tchem like function to compute temperature on host
     * @param policy
     * @param state
     * @param internalEnergyRef
     * @param enthalpyMass
     * @param temperature
     * @param kmcd
     */
    [[maybe_unused]] static void runHostBatch(  /// thread block size
        typename UseThisTeamPolicy<host_exec_space>::type& policy,
        /// input
        const real_type_2d_view_host_type& state,
        /// output
        const real_type_1d_view_host_type& speedOfSound,
        /// const data from kinetic model
        const kinetic_model_host_type& kmcd);
};

}  // namespace ablate::eos::tChem
#endif
