# DRV-300 Variable Frequency Drive Service Manual

Capture the drive event history, requested speed and operating load before
clearing alarms. Fault-code meanings below apply to the DRV-300 sample drive.

## VFD-OC3 acceleration overcurrent

VFD-OC3 is an acceleration overcurrent trip. Record the current trace, commanded
acceleration time and whether the driven equipment can move freely after the
approved isolation procedure. An unusually short ramp or a mechanical jam can
produce high acceleration current. Recurring trips at unchanged settings call
for review of the motor and cable by qualified electrical maintenance.

### Distinguishing OC3 from a steady-speed trip

Use the event timestamp and speed trace to determine where the trip occurred.
An acceleration event points toward starting load and ramp settings. A trip at
steady speed calls for load, motor and cable checks. Preserve both traces if the
event timing is ambiguous.

## VFD-UV1 DC bus undervoltage

VFD-UV1 indicates a low DC bus event. Correlate the drive history with upstream
supply interruptions and input voltage records. A site-wide voltage dip can
affect other drives at the same timestamp.

## E01.2 cooling fan feedback

E01.2 indicates missing cooling fan feedback. Capture the event history and
observe cooling performance through the maintenance procedure. Escalate a
repeating feedback fault to drive service.
