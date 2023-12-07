---
layout: default
title: Configuration Files
parent: Installation
nav_order: 4
---
## Prebuilt Configuration Files
ABLATE builds some configuration files that may be useful for building ABLATE, PETSc, or TensorFlow on specific machines.  See the associated [BuildWiki](https://github.com/UBCHREST/ablate/wiki) for details.

### Configuration Files List
{% for file in site.static_files %}
{% if file.path contains 'content/installation/config' %}
- [{{ file.path | remove: '/content/installation/config/'}}]({{ file.path }})
{% endif %}
{% endfor %}