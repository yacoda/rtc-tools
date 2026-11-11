#!/bin/bash
# omc of the OpenModelica rtc-integration branch (OM_INTEGRATION), run in the shell_compiler:master
# container it was built in. /tmp and the home folder are mounted at their own paths, so that the
# paths in the export scripts of rtc-tools work unchanged; the Modelica Standard Library comes
# from ~/.openmodelica/libraries.
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp/omc-home \
  -e OPENMODELICALIBRARY="$HOME/.openmodelica/libraries" \
  -v "${OM_INTEGRATION:?}":/local:ro -v /tmp:/tmp -v "$HOME":"$HOME":ro \
  -w "$(pwd)" -e PATH=/local/build/install_cmake/bin:/usr/local/bin:/usr/bin:/bin \
  shell_compiler:master omc "$@"
