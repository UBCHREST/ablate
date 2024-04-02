---
layout: default
title: Zero-RK
parent: Chemistry
nav_order: 3
grand_parent: Formulations
---

Ideal gas thermal properties and reactions kinetics are modeled using the [Zero-RK](https://github.com/LLNL/zero-rk) tool kit. 
The input files needs to be using the Chemkin formated files, one .inp file containing the reaction and one .dat file containing thermodynamic properties.
The mech.log file provides useful information when parsing of the mechanism file is not successful.
The default chemistry Integrator is CVODE with an optional integrator, SEULEX, for stiff chemistry problems.
In order to try to limit the stiffness, the steplimiter option can be reduced. 
For 1D flames significant behavioral change can be observed with stepLimiter<1E18.
The load balancing option should always be turned on for any 1D,2D or 3D simulation.
Reactors with initial temperatures below thresholdTemperature will not be evaluated.
When an approximate jacobian or when sparseJacobian is used set iterative option to 1.
Zero-RK has an option to take advantage of sparse jacobian for the reactor evaluations.
For 0D cases it has been hwn to have improved performance over dense jacobian evaluations for mechanism larger than about 100 species.

Examples of specifying zerorkEOS with different default options and possible options.

```yaml
   # relative path to chemkin input files
    - eos: &eos !ablate::eos::zerorkEOS
        reactionFile: ../mechanisms/grimech30.mech.inp
        thermoFile: ../mechanisms/grimech30.thermo.dat

        options:
          # double options
          relTolerance: 1.0E-6
          absTolerance: 1.0E-10
          thresholdTemperature: 0 # use 560 for reduced PMMA mechanism
          stepLimiter: 1.0E200          
          
          # integer options
          loadBalance: 1 # [0 or 1 or 2] 1 is based on #of reactors 2 is based on time
          verbose: 0 # [0 or 1 or 2 or 3 or 4]
          gpu: 0 # [0 or 1]
          itterative: 0 # [0 or 1]
          useSeulex: 0 # [0 or 1]  default is CVODE     
          
          # boolean options:
          sparseJacobian: false #default is dense
          timingLog: false

          # Reactor types:
          reactorType: ConstantVolume # [ConstantPressure or ConstantVolume]


```
