# Hardware

Labels as in the plan. **(sec)** = secondary source (reseller/aggregator), to be replaced by a vendor datasheet in Phase 10. **(conflict)** = sources disagree.

## 1. Arms (decision D-29: 2 × Franka Research 3)

| attribute | Franka Research 3 (chosen) | Kinova Gen3 7-DoF | Flexiv Rizon 4 / 4s | UR e-Series |
|---|---|---|---|---|
| DoF | 7 [V:S-23] | 7 [V:S-24] | 7 [V:S-25] | 6 (sec) |
| Payload | 3 kg | 4 kg (sec: 2 kg continuous) | 4 kg (4s sources differ) | 5 kg UR5e (sec) |
| Reach | 855 mm | 902 mm | 780 vs 876 mm **(conflict)**; 4s 919 mm | ~850 mm (sec) |
| Repeatability | < ±0.1 mm | ±0.1 mm (sec) | ±0.05 mm | ±0.03 mm (sec) |
| Joint torque sensing | all 7 joints, link-side [V:S-23] | in actuators [V:S-24] | per joint [V:S-25] | none (sec) |
| Real-time interface | FCI 1 kHz torque/joint/Cartesian, open-source `libfranka` [V:S-23] | 1 kHz low-level incl. torque; `ros2_kortex` [V:S-24] | vendor RDK **[G]** | RTDE **[G]** |
| Impedance references | Cartesian impedance + dual-arm examples published [V:S-20, S-21] | **[G]** | **[G]** | external F/T needed |
| Safety documentation | claims EN ISO 13849-1 functions (sec) | not established | 18 safety functions to EN ISO 13849-1 Cat 3 PL d [V:S-25] | not established |
| Mass | 17.8 kg (reseller) | 8.2 kg | ~20 kg | — |
| Price | ≈ $29k (reseller) | quote | quote | quote |

Why FR3: the project's central behaviour (§4 of the plan) needs a torque-controlled arm with joint torque sensing; FR3 is the one whose impedance controllers are published and whose torque interface runs at 1 kHz in open source. Kinova Gen3 and Rizon are drop-in alternatives behind the L1 interface. FR3 payload (3 kg incl. gripper) restricts objects **[L]**. Franka also offers the **FR3 Duo** reference setup (two FR3 on a dual-arm bracket with a sensor head) with CAD/URDF/USD/MJCF models [V:S-32]; its bracket geometry is the baseline for the reachability study, and purchase availability is confirmed at quoting. The sim model version (FR3 v1 / v2 / v2.1) must equal the purchased arm's version.

## 2. End-effector and sensing
* Franka Hand (default). Wrist F/T not required on FR3 (joint-torque wrench estimate); evaluated only if G-10/hardware push tests show the estimate is too noisy for `F_push` detection.
* Mounting: rigid steel/aluminium table or frame, base spacing from the P2 reachability study, load calculation from the datasheet base-torque limits **[G]**.

## 3. Eyes
| piece | Eye-A (default) | Eye-B (fallback) |
|---|---|---|
| camera | ONVIF PTZ IP camera, optical zoom, RTSP [V:S-29] | machine-vision GigE/USB3 camera **[G: part chosen in P11]** |
| pan/tilt | in-camera PTZ | servo pan/tilt head (Dynamixel-class) **[G]** |
| ROS 2 | `axis_camera`-style PTZ driver / `ptz_action_server_msgs` precedent [V:S-29] | custom `ros2_control` |
| rail | belt/rack carriage on ring/arc, stepper/servo + encoder, homing + limit switches (custom build; no commercial part verified) | same |
| calibration | 12 stations × zoom levels, AprilTag boards on table/frames | same |
Gate G-9: pan/tilt return error ≤ 0.1°, zoom intrinsics reproducible; fail → Eye-B.

## 4. Compute and safety peripherals
RT-PC (PREEMPT_RT, wired link per arm) running L1 only; Brain/vision workstation separate; PTP/chrony sync; hard-wired E-stop and enabling device; safety laser scanner or curtain for speed-and-separation monitoring (model selected in P10 **[G]**).
