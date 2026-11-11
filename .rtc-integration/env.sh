# Source this: the environment of the rtc-integration branches, for the examples and tests/routes.
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INTEGRATION=${INTEGRATION:-$(cd "$_here/../.." && pwd)}
export INTEGRATION
source "${VENV:-$INTEGRATION/venv}/bin/activate"
# omc of the OpenModelica rtc-integration branch, in its build container (omc.sh)
export RTCTOOLS_OMC="$_here/omc.sh"
export OM_INTEGRATION=${OM_INTEGRATION:-$(cd "$INTEGRATION/../OpenModelica/om-integration" && pwd)}
export RTCTOOLS_RUMOCA="$INTEGRATION/rumoca/target/release/rumoca"
# lacemodelica (yacoda/lacemodelica main and the rtc-tools frontend branch)
export PATH="${LACEMODELICA_BIN:-/home/yacoda/programs/lacemodelica-rtc/build}:$PATH"
unset _here
