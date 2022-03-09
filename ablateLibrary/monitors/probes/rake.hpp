#ifndef ABLATELIBRARY_RAKE_HPP
#define ABLATELIBRARY_RAKE_HPP

#include "probeInitializer.hpp"
#include <string>
#include <vector>

namespace ablate::monitors::probes {

class Rake : public ProbeInitializer{
   private:
    //! the name of these rakes/probes
    const std::string rakeName;

    //! the base path to the probe directory
    const std::filesystem::path rakePath;

    //! The list of probes
    std::vector<Probe> list;

   public:
    /**
     * default list of probe monitors, useful for the parser init
     * @param probes
     */
    explicit Rake(std::string name, std::vector<double> start, std::vector<double> end, int number);

    /**
     * list of probe locations with names
     * @return
     */
    const std::vector<Probe>& GetProbes() const override { return list; }


    /**
     * place all probes in the same directory
     * @return
     */
    std::filesystem::path GetDirectory() const override { return rakePath; }

    /**
     * prints a log of probe locations to a file
     */
    void Report(MPI_Comm) const override;

};

}
#endif  // ABLATELIBRARY_RAKE_HPP
