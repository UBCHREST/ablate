#ifndef ABLATELIBRARY_PROBES_HPP
#define ABLATELIBRARY_PROBES_HPP

#include <utility>
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "probes/probe.hpp"
#include "probes/probeInitializer.hpp"

namespace ablate::monitors {

/**
 * Used to monitor field variables at a specified list of locations
 */
class Probes : public Monitor {
   public:
    /**
     * Private class for recording the the probe output
     */
    class ProbeRecorder {
       private:
        //! The amount of data to store before writing
        const int bufferSize;

        //! The output path for the csv file
        std::filesystem::path outputPath;

        //! The last output, useful for restart
        PetscReal lastOutputTime = PETSC_MIN_REAL;

        //! The current location in the buffer to record
        int activeIndex = -1;

        //! store the output buffer [buffer][variables]
        std::vector<std::vector<double>> buffer;

        //! store the time history
        std::vector<double> timeHistory;

       public:
        /**
         * List of variables to output in the order that they will be set
         * @param bufferSize
         * @param variables
         */
        ProbeRecorder(int bufferSize, const std::vector<std::string>& variables, const std::filesystem::path& outputPath);

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
    const std::shared_ptr<ablate::monitors::probes::ProbeInitializer> initializer;

    //! List of variables to output
    const std::vector<std::string> variableNames;

    //! The sampling interval
    const std::shared_ptr<io::interval::Interval> interval;

    //!  output bufferSize
    const int bufferSize;

    //! list of local probes on this rank
    std::vector<probes::Probe> localProbes;

    //! list of fields to interpolate
    std::vector<domain::Field> fields;

    //! store the offset for the field in the output (needed for multiple components)
    std::vector<int> fieldOffset;

    //! list of petsc intepolants
    std::vector<DMInterpolationInfo> interpolants;

    //! list of probe recorders that goe
    std::vector<ProbeRecorder> recorders;

    static PetscErrorCode UpdateProbes(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);

   public:
    /**
     * Probes monitor
     * @param probeLocations a vector of names and probe locations
     * @param variables a list of output variables
     * @param bufferSize the buffer size between writes
     * @param interval the sampling interval
     */
    Probes(const std::shared_ptr<ablate::monitors::probes::ProbeInitializer>&, std::vector<std::string> variableNames, const std::shared_ptr<io::interval::Interval>& interval = {},
           const int bufferSize = 0);

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
