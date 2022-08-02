---
layout: default
title: Formatting
parent: Code Development
nav_order: 5
---

The c++ code style is based upon the [Google Style Guide](https://google.github.io/styleguide/) and enforced using clang-format during PR tests.  Specific overrides to the style are controlled in the .clang-format file.

```bash
# To run a format check
# from build directory
make format-check
```

It is recommended that an IDE (e.g. CLion) is used to check formatting during development.  Clang-format can also be used during development to directly format each file.
```bash
clang-format -i path/to/file
```
