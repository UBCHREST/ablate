
#ifndef ABLATELIBRARY_PROBES_HPP
#define ABLATELIBRARY_PROBES_HPP

#include <particles/particles.hpp>
#include <utility>
#include "io/interval/interval.hpp"

namespace ablate::monitors {

/**
 * Used to monitor field variables at a specified list of locations
 */
class Probes : public Monitor {
   public:
    struct Probe {
        std::string name;
        std::vector<PetscReal> location;

        Probe(std::string name, std::vector<PetscReal> location) : name(std::move(name)), location(std::move(location)) {}
    };

    /**
     * Private class for recording the the probe output
     */
    class ProbeRecorder{
       private:
        //! The amount of data to store before writing
        const int bufferSize;

        //! The output path for the csv file
        std::filesystem::path outputPath;

        //! The last output, useful for restart
        PetscReal lastOutputTime = PETSC_MIN_REAL;

        //! The current location in the buffer to record
        std::size_t activeIndex = -1;

        //! store the output buffer [buffer][variables]
        std::vector<std::vector<double>> buffer;

       public:
        /**
         * List of variables to output in the order that they will be set
         * @param bufferSize
         * @param variables
         */
        ProbeRecorder(int bufferSize, const std::vector<std::string>& variables, std::filesystem::path outputPath);

        /**
         * Catch close and output the buffer
         */
        ~ProbeRecorder();

        /**
         * Advance and record the next time.  Output the buffer if needed
         * @param time
         */
        void AdvanceTime(double time);

        /**
         * Advance and record the the value at the current time
         * @param time
         */
        void SetValue(std::size_t index, double value);

        /**
         * Writes and resets the buffer
         */
        void WriteBuffer();
    };

    //! Original list of all requested probe locations by name
    const std::vector<Probe> allProbes;

    //! List of variables to output
    const std::vector<std::string> variableNames;

    //! The sampling interval
    const std::shared_ptr<io::interval::Interval> interval;

    //! list of local probes on this rank
    std::vector<Probe> localProbes;

    //! list of fields to interpolate
    std::vector<domain::Field> fields;

    //! list of petsc intepolants
    std::vector<DMInterpolationInfo> interpolants;

    static PetscErrorCode UpdateProbes(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);

   public:
    /**
     * Probes monitor
     * @param probeLocations a vector of names and probe locations
     * @param variables a list of output variables
     * @param bufferSize the buffer size between writes
     * @param interval the sampling interval
     */
    Probes(std::vector<Probe> probes, std::vector<std::string> variableNames, const std::shared_ptr<io::interval::Interval>& interval = {});

    /**
     * Probes monitor
     * @param probeLocations a vector of names and probe locations as shared pointers.  This is used for the parser
     * @param variables a list of output variables
     * @param bufferSize the buffer size between writes
     * @param interval the sampling interval
     */
    Probes(std::vector<std::shared_ptr<Probe>> probes, std::vector<std::string> variableNames, const std::shared_ptr<io::interval::Interval>& interval = {});

    ~Probes() override;

    /**
     * Overrides the base register and used to determine which nodes lives on each rank
     * @param solver
     */
    void Register(std::shared_ptr<solver::Solver> solver) override;

    /**
     * Returns the petsc function used to update the monitor.
     * @return
     */
    PetscMonitorFunction GetPetscFunction() override { return UpdateProbes; }
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_PROBES_HPP
