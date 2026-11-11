#!/bin/bash
# Export the example models as plain FMI 3.0 Model Exchange FMUs: no --daeMode, symbolic ODE Jacobian,
# no tearing. OpenModelica needs the FMU interface of its branch fmu-ode-mode.
#
# Wrappers: restate the initial equations as outputs _INITres<i>, which FMUMixin reads in ODE mode, since a
# plain FMU does not expose them; literal attributes, as in the FMUs of the DAE route; start values
# where the FMU cannot evaluate 0. Models that OpenModelica refuses as under-determined get their
# free variable as an input, and delay() becomes an input for delayed_feedback().
#
# usage: OMC=<command that runs omc in the current folder> WORK=<scratch folder>
#        EXAMPLES=<rtc-tools/examples> MSL=<Modelica 4 folder> CHANNEL_FLOW=<folder with Deltares/>
#        export_fmus.sh
# Output: $WORK/fmus/<example>/<name>.fmu
: "${OMC:=omc}" "${WORK:?}" "${EXAMPLES:?}" "${MSL:?}" "${CHANNEL_FLOW:?}"
OPTIONS="--fmiInputDerivatives -d=-disableDirectionalDerivatives --generateDynamicJacobian=symbolic --tearingMethod=noTearing"
mkdir -p "$WORK/open"
export_fmu() {
  local example=$1 output=$2 class=$3 wrapper=$4; shift 4
  local work=$WORK/exp/$example-$output
  rm -rf "$work"; mkdir -p "$work" "$WORK/fmus/$example"
  {
    echo "loadFile(\"$MSL/package.mo\"); getErrorString();"
    echo "loadFile(\"$CHANNEL_FLOW/Deltares/package.mo\"); getErrorString();"
    for file in "$@"; do case $file in /*) echo "loadFile(\"$file\"); getErrorString();";; *) echo "loadFile(\"$EXAMPLES/$example/model/$file\"); getErrorString();";; esac; done
    if [ -n "$wrapper" ]; then
      echo "$wrapper" > "$work/wrapper.mo"
      echo "loadFile(\"$work/wrapper.mo\"); getErrorString();"
    fi
    echo "setCommandLineOptions(\"$OPTIONS\"); getErrorString();"
    echo "buildModelFMU($class, version=\"3.0\", fmuType=\"me\", fileNamePrefix=\"$class\"); getErrorString();"
  } > "$work/export.mos"
  (cd "$work" && $OMC export.mos) > "$work/omc.log" 2>&1
  if [ -f "$work/$class.fmu" ]; then
    cp "$work/$class.fmu" "$WORK/fmus/$example/$output.fmu"; echo "OK $example/$output"
  else
    echo "FAILED $example/$output: $(grep -m2 -E 'Error' "$work/omc.log" | grep -v index.json | cut -c1-200)"
  fi
}
# init_res <expr>...: scalar outputs _INITres<i> restating the initial equations as residuals
init_res() { local i=0 e; for e in "$@"; do i=$((i + 1)); printf 'output Real _INITres%d = %s; ' "$i" "$e"; done; }
q_res() { local c=$1 from=$2 to=$3 i; for ((i = from; i <= to; i++)); do echo "$c.Q[$i] - Inflow_Q"; done; }
mapfile -t R < <(q_res Channel 2 11); CP=$(init_res "${R[@]}")
mapfile -t R < <(q_res Channel 2 2); SR=$(init_res "${R[@]}")
export_fmu basic Example Example "" Example.mo &
export_fmu single_reservoir SingleReservoir SingleReservoir "" single_reservoir.mo &
export_fmu step_response Example Example_fmu "model Example_fmu extends Example(Channel(H(each min = 0.2, each start = 3.0, each stateSelect = StateSelect.prefer), Q(each nominal = 100))); $SR end Example_fmu;" Example.mo &
export_fmu step_response Example_convective Example_fmu_convective "model Example_fmu_convective extends Example(Channel(use_convective_acceleration = true, H(each min = 0.2, each start = 3.0, each stateSelect = StateSelect.prefer), Q(each nominal = 100))); $SR end Example_fmu_convective;" Example.mo &
wait
export_fmu channel_pulse Example Example_fmu "model Example_fmu extends Example(Channel(H(each min = -5.0, each stateSelect = StateSelect.prefer), Q(each nominal = 100, each start = 100))); $CP end Example_fmu;" Example.mo &
export_fmu channel_pulse Example_convective Example_fmu_convective "model Example_fmu_convective extends Example(Channel(use_convective_acceleration = true, H(each min = -5.0, each stateSelect = StateSelect.prefer), Q(each nominal = 100, each start = 100))); $CP end Example_fmu_convective;" Example.mo &
export_fmu channel_pulse Example_upwind Example_fmu_upwind "model Example_fmu_upwind extends Example(Channel(use_convective_acceleration = true, use_upwind = true, H(each min = -5.0, each stateSelect = StateSelect.prefer), Q(each nominal = 100, each start = 100))); $CP end Example_fmu_upwind;" Example.mo &
reaches='upstream(H(each min = 15, each start = 20, each stateSelect = StateSelect.prefer), Q(each nominal = 100)), middle(H(each min = 10, each start = 15, each stateSelect = StateSelect.prefer), Q(each nominal = 100)), downstream(H(each min = 5, each start = 10, each stateSelect = StateSelect.prefer), Q(each nominal = 100))'
mapfile -t R < <(q_res downstream 2 5; q_res middle 2 5; q_res upstream 2 5); WO=$(init_res "${R[@]}"); WL=$(init_res "${R[@]}" upstream_pid.error_integral middle_pid.error_integral)
export_fmu channel_wave_damping ExampleOptimization ExampleOptimization_fmu "model ExampleOptimization_fmu extends ExampleOptimization($reaches); $WO end ExampleOptimization_fmu;" Example.mo ExampleOptimization.mo PIDController.mo &
wait
export_fmu channel_wave_damping ExampleLocalControl ExampleLocalControl_fmu "model ExampleLocalControl_fmu extends ExampleLocalControl($reaches); $WL end ExampleLocalControl_fmu;" Example.mo ExampleLocalControl.mo PIDController.mo &
mapfile -t R < <(for c in LowerChannel MiddleChannel UpperChannel; do echo "der($c.Q[2])"; echo "der($c.Q[3])"; done); CC=$(init_res "${R[@]}")
export_fmu cascading_channels Example Example_fmu "model Example_fmu extends Example(LowerChannel(H(min = {-1.5, -1.5})), MiddleChannel(H(min = {-1.0, -1.0})), UpperChannel(H(min = {-0.5, -0.5}))); input Real Level_H(min = -1.5, max = 1.0); $CC equation Level.H = Level_H; end Example_fmu;" Example.mo &
export_fmu ensemble Example Example "" Example.mo &
export_fmu goal_programming Example Example "" Example.mo &
wait
python3 - "$WORK/open/IntegratorDelay.mo" "$EXAMPLES/integrator_delay/model/Example.mo" <<'PY'
import re, sys
s = open(sys.argv[2]).read()
s = re.sub(r"Deltares\.ChannelFlow\.SimpleRouting\.Branches\.Delay delay_component\([^)]*\)", "OpenDelay delay_component", s)
s = s.replace("equation\n", "  input Modelica.Units.SI.VolumeFlowRate Q_delayed;\nequation\n  delay_component.QOut.Q = Q_delayed;\n", 1)
s = "block OpenDelay\n  extends Deltares.ChannelFlow.Internal.QSISO;\nend OpenDelay;\n" + s
open(sys.argv[1], "w").write(s)
PY
export_fmu integrator_delay Example Example "" "$WORK/open/IntegratorDelay.mo" &
export_fmu lookup_table Example Example "" Example.mo &
export_fmu mixed_integer Example Example "" Example.mo &
sed 's/^  Boolean Turbine_is_on;/  input Boolean Turbine_is_on;/' $EXAMPLES/pumped_hydropower_system/model/PumpedStoragePlant.mo > "$WORK/open/PumpedStoragePlant.mo"
export_fmu pumped_hydropower_system PumpedStoragePlant PumpedStoragePlant "" "$WORK/open/PumpedStoragePlant.mo" &
wait
