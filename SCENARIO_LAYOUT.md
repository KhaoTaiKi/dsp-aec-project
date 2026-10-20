# Scenario Layout: Static and Moving Echo Experiments

The project uses the same signal convention in every scenario:

- `farend`: voice played through the simulated loudspeaker; adaptive-filter reference.
- `nearend`: separate voice close to the simulated microphone; absent in S1.
- `mic`: simulated microphone mixture.
- `echo_estimate`: echo predicted from `farend`.
- `residual`: `mic - echo_estimate`.

## Scenario progression

| ID | Speakers | Room path | Simulated microphone | Purpose |
| --- | --- | --- | --- | --- |
| S1 | One far-end voice | One fixed RIR | `farend * rir_0` | Verify echo cancellation. Residual should approach silence. |
| S2 | Far-end + near-end voice | One fixed RIR | `(farend * rir_0) + nearend` | Cancel echo while retaining the near-end voice. |
| S3 | One or two voices | RIR switches at declared times | `farend * rir(t) [+ nearend]` | Test tracking when a source or microphone changes position. |

## Folder contract

```text
configs/scenarios/        Versioned experiment settings; no audio.
manifests/                Versioned CSV/JSON index of clips, RIRs, segments, and switch times.
data/source_clips/        Local clean voice clips; ignored by Git.
data/simulated/<scenario>/ Generated mic/reference/target WAV files; ignored by Git.
results/<scenario>/       Tables and figures produced by a scenario.
src/                      Shared simulation, adaptive-filter, metric, and runner code.
```

## Required manifest fields

Every generated trial must record: `scenario_id`, `trial_id`, `sample_rate`, `farend_clip`, `nearend_clip` (empty for S1), `rir_id`, `rir_switch_samples`, `segment_start`, `segment_end`, `mic_path`, `reference_path`, `target_path`, and `seed`.

For S3, `rir_switch_samples` is mandatory. A movement case is represented by switching from one known RIR to another at a recorded sample index; no position change may be implicit or undocumented.

## Execution order

Only S1 is runnable now. S2 and S3 configs are placeholders that define compatible inputs and outputs. They must use the same sample rate, manifest schema, output names, and metric pipeline when activated.
