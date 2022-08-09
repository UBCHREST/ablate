---
layout: default
title: TChem
parent: Chemistry
nav_order: 3
---

[![TChem Logo](https://github.com/sandialabs/TChem/blob/main/logo/TChem_Logo_Large.png?raw=true)](https://github.com/sandialabs/TChem)

Ideal gas thermal properties and reactions kinetics are modeled using the [TChem](https://github.com/sandialabs/TChem) tool kit.  TChem is build automatically when configuring ABLATE and can be specified using either Chemkin or [Cantera (yaml)](https://cantera.org/documentation/dev/sphinx/html/yaml/index.html) mechanism files. When using the TChem equation of state the order of the species solved is dictated by the order specified in the mechanism file where the last species is computed to sum to unity.  Examples of specifying the TChem EOS are provided below.

```yaml
   # relative path to chemkin input files
    - eos: &eos !ablate::eos::TChem
        mechFile: ../mechanisms/grimech30.mech.dat
        thermoFile: ../mechanisms/grimech30.thermo.dat

   # relative path to cantera input files
    - eos: &eos !ablate::eos::TChem
        mechFile: ../mechanisms/grimech30.yaml

   # web located cantera input file
    - eos: &eos !ablate::eos::TChem
        mechFile: !ablate::environment::Download
          https://raw.githubusercontent.com/UBCHREST/ablate/main/tests/integrationTests/inputs/mechanisms/gri30.yaml
```
