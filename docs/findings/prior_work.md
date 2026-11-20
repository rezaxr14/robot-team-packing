# Prior work — language-coordinated multi-arm teams, attribution, memory, verification, safe interaction

**Scope.** Research question posed by the main session: how others have built and evaluated language-coordinated multi-arm robot teams with typed messages, verifiers, memory and tool use, and what this implies for our plan. Context read first: `paper/sections/01_intro.tex`, `paper/sections/02_related.tex`, `docs/AGENT_BUILD_SPEC.md`, `docs/DATAGEN_KAGGLE.md`, `docs/USER_PLAN_VERBATIM.md`.

**Method and limits (honest).** Gathered on 2026-11-20. `websearch` returned HTTP 402 on every call, so discovery was done through the public arXiv API (`export.arxiv.org/api/query`), OpenAlex and Crossref via `webfetch`. **I read titles, authors, year, abstracts and venue/comment metadata — not full texts.** Peer-review status is taken from arXiv `comment`/`journal_ref` or OpenAlex `primary_location`; where the record did not state a venue I marked it *preprint* (which may understate publication). Several sources are 2026 preprints and may be under review. Every source below is an **external source, not reproduced here**; no number from it is a result of this repository. 32 sources total.

---

## 1. Published systems that coordinate multiple arms/embodied agents with LM agents

**S1. RoCo: Dialectic Multi-Robot Collaboration with Large Language Models** — Zhao Mandi, Shreeya Jain, Shuran Song (Columbia University); 2023; https://arxiv.org/abs/2307.04738.
- Claim: one LLM per arm discusses a task in natural language, emits subtask plans plus task-space waypoints that a multi-arm motion planner grounds, with collision-checking fed back into the dialogue in-context; RoCoBench covers 6 tasks and includes real-world human-in-the-loop trials.
- Status: preprint (venue not stated in arXiv record). *External source, not reproduced here.*

**S2. Building Cooperative Embodied Agents Modularly with Large Language Models (CoELA)** — Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, Chuang Gan (UMass Amherst, Tsinghua, SJTU, MIT, IBM); 2023 (ICLR 2024); https://arxiv.org/abs/2307.02485.
- Claim: a modular LLM agent separates perception, memory, communication, planning and execution, works under costly communication and raw sensory input, and GPT-4-driven agents beat planning baselines on C-WAH/TDW-MAT while exhibiting emergent communication; self-collected fine-tuning data helps open models.
- Status: peer-reviewed (ICLR 2024). *External source, not reproduced here.*

**S3. VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning** — Li Kang, Xiufeng Song, Heng Zhou, Yiran Qin, Jie Yang, Xiaohong Liu, Philip Torr, Lei Bai, Zhenfei Yin (Shanghai Jiao Tong University, Oxford, Shanghai AI Lab); 2025; https://arxiv.org/abs/2506.09049.
- Claim: VIKI-Bench is a hierarchical 3-level benchmark (agent activation, task planning, trajectory perception) with multi-view observations, and VIKI-R fine-tunes a VLM on CoT demonstrations then applies RL with multi-level rewards, reporting gains over baselines and emergent compositional cooperation.
- Status: peer-reviewed (NeurIPS 2025 Datasets & Benchmarks). *External source, not reproduced here.*

**S4. NSDL: Neuro-Symbolic Division of Labor for Multi-Robot Task Planning** — Xincheng Tan, Ke Wang, Yining Di, Minfeng Zhu; 2026; https://arxiv.org/abs/2611.16671.
- Claim: a small VLM writes a "plan sketch" of scene judgments and intends steps while a deterministic solver expands it into an executable joint plan, with detector and solver as tools and the solver in the training loop; on VIKI-L2 and the newly public LOCUS-Bench the 7B model reaches 97.0% OOD accuracy, ordering/scheduling errors "nearly disappear" and residual failures are misjudged states.
- Status: preprint (submitted to ICRA 2027; code/weights/LOCUS-Bench public at https://github.com/Xincheng-Tan/NSDL). *External source, not reproduced here.*

**S5. LAN2CB (Compositional Coordination for Multi-Robot Teams with Large Language Models)** — Zhehui Huang, Guangyao Shi, Yuwei Wu, Vijay Kumar, Gaurav S. Sukhatme (University of Pennsylvania, USC); 2025; https://arxiv.org/abs/2507.16068.
- Claim: an NL mission is parsed into behavior trees and then compiled to executable Python control code using a structured knowledge base, reducing manual expert translation and generalizing across mission types in simulation and on real robots.
- Status: peer-reviewed (IEEE MRS 2025, oral). *External source, not reproduced here.*

**S6. Transforming Monolithic Foundation Models into Embodied Multi-Agent Architectures for Human-Robot Collaboration (InteractGen)** — Nan Sun, Bo Mao, Yongchang Li, Chenxu Wang, Di Guo, Huaping Liu (Tsinghua University, BUPT); 2025 preprint / 2026 journal version; https://arxiv.org/abs/2512.00797 · https://doi.org/10.1109/tro.2026.3751844.
- Claim: robot intelligence is decomposed into specialized agents for continuous perception, dependency-aware planning, decision *and verification*, failure reflection and dynamic human delegation as "regulated components" of a closed-loop collective, evaluated on a heterogeneous team in a three-month open-use study.
- Status: peer-reviewed (IEEE Transactions on Robotics 2026, per OpenAlex/Crossref DOI). *External source, not reproduced here.*

**S7. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation** — Tianxing Chen et al. (>25 authors); 2025; https://arxiv.org/abs/2506.18088.
- Claim: a dual-arm data generator with a 731-instance/147-category object library, MLLM-plus-simulation-in-the-loop task-code synthesis and 5-axis domain randomization across 50 tasks and 5 embodiments reports +10.9% code-generation success and a VLA trained on synthetic data plus 10 real demonstrations improving 367% over a 10-demo baseline.
- Status: preprint with public code/project page (earlier RoboTwin 1.0 is CVPR 2025 Highlight, https://arxiv.org/abs/2504.13059). *External source, not reproduced here.*

**S8. From Vision to Harvest: Benchmarking Vision-Language Models for Multi-Arm Robotic Fruit Harvesting** — Vrishan Inukollu, Adyan Zaman, Anvi Kudaraya, Carlos Lazcano, Yuankai Zhu, Stavros Vougioukas, Xiaofan Yu; 2026; https://arxiv.org/abs/2609.13606.
- Claim: frontier VLMs generate zero-shot multi-arm harvesting sequences and waypoints checked by a lightweight trajectory verifier, but practical deployment is limited by accurate 3D waypoint generation and collision-aware coordination.
- Status: preprint. *External source, not reproduced here.*

**Synthesis.** The recurring split is perception → language planning/communication → grounded execution, with a motion planner or solver as the authority on feasibility (S1, S4, S5) or with verification as an explicit role (S6, S8). Communication is mostly free-form natural-language dialogue (S1, S2, S6), not typed messages with schemas; **none of the records I read claims typed, budgeted messages with per-message replayable claims like our SCENE/PLAN bus**. S4 is the closest competitor to our "Brain emits skills + deterministic dry run" design and directly overlaps the paper's open TODO about LOCUS-Bench (which is now public, contrary to `paper/sections/02_related.tex:20`).

---

## 2. Making failures attributable (counterfactual replay, causal attribution, hindsight, credit)

**S9. Counterfactual Multi-Agent Policy Gradients (COMA)** — Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, Shimon Whiteson (Oxford); 2017; https://arxiv.org/abs/1705.08926.
- Claim: a centralized critic with decentralized actors uses a counterfactual baseline that marginalizes out one agent's action while holding the others fixed, computed in one forward pass, improving multi-agent credit assignment on StarCraft micromanagement.
- Status: peer-reviewed (AAAI 2018; venue not in arXiv record). *External source, not reproduced here.*

**S10. Hindsight Experience Replay (HER)** — Marcin Andrychowicz, Filip Wolski, Alex Ray et al. (OpenAI); 2017; https://arxiv.org/abs/1707.01495.
- Claim: relabeling a failed trajectory with the goal it actually reached enables sample-efficient learning from binary sparse rewards on pushing/sliding/pick-and-place and transfers from simulation to a physical arm.
- Status: peer-reviewed (NeurIPS 2017; venue not in arXiv record). *External source, not reproduced here.*

**S11. Difference Rewards Policy Gradients (Dr.Reinforce)** — Jacopo Castellini, Sam Devlin, Frans A. Oliehoek, Rahul Savani; 2020 (journal 2022); https://arxiv.org/abs/2012.11258.
- Claim: directly differencing a known reward function yields per-agent credit without learning a Q-function as COMA does, with a learned reward-network variant when the reward is unknown.
- Status: peer-reviewed (Neural Computing & Applications 2022; AAMAS 2021 extended abstract). *External source, not reproduced here.* (This resolves the "confirm difference-rewards literature" TODO in `paper/sections/02_related.tex:13`.)

**S12. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems (Who&When)** — Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, Qingyun Wu; 2025; https://arxiv.org/abs/2505.00212.
- Claim: on failure logs from 127 LLM multi-agent systems the best automated method reaches only 53.5% accuracy for the responsible agent and 14.2% for the decisive step, with some methods below random and o1/R1 not practically usable.
- Status: camera-ready (venue not stated in arXiv record; treat as preprint/camera-ready). *External source, not reproduced here.*

**S13. Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems (TraceElephant)** — Mengzhuo Chen, Junjie Wang, Fangwen Mu, Yawen Wang, Zhe Liu, Huanxiang Feng, Qing Wang; 2026; https://arxiv.org/abs/2604.22708.
- Claim: attribution should be studied with full execution traces (agent inputs and context, not only outputs), and full traces improve attribution accuracy by up to 76% over partial observations.
- Status: peer-reviewed (ACL 2026). *External source, not reproduced here.*

**S14. Adaptive Influence Graphs for Failure Attribution in Multi-Agent Systems** — Yarden Bakish, Amir Dudai, Roy Ganz, Oren Nuriel, Elad Ben Avraham, Mor Shpigel Nacson, Ron Litman; 2026; https://arxiv.org/abs/2608.24361.
- Claim: transforming a failed trace into a structured influence graph and navigating it agentically sets a new state of the art on Who&When, showing that trace representation matters as much as the diagnosing model.
- Status: preprint. *External source, not reproduced here.*

**S15. Rethinking Failure Attribution in Multi-Agent Systems: A Multi-Perspective Benchmark and Evaluation (MP-Bench)** — Yeonjun In, Mehrab Tanjim, Jayakumar Subramanian, Sungchul Kim, Uttaran Bhattacharya, Wonjoong Kim, Sangwu Park, Somdeb Sarkhel, Chanyoung Park; 2026; https://arxiv.org/abs/2603.25001.
- Claim: MAS failures often admit several plausible attributions, and prior conclusions that LLMs are bad at attribution are largely driven by benchmarks that assume a single deterministic root cause.
- Status: preprint (under review). *External source, not reproduced here.*

**S16. What Would Removing This Corruption Change? Failure Attribution for Vision-Language-Action Models** — Carson Sobolewski, Anjali Parashar, Fei Chen, Yingke Li, Chuchu Fan (MIT); 2026; https://arxiv.org/abs/2610.27277.
- Claim: because measuring the benefit of removing a corruption requires costly paired counterfactual rollouts, a factorized risk model estimates removal benefit from the current image and instruction without them, reaching ~3x lower attribution error than frontier VLMs on OpenVLA and π0.5 (including π0.5-DROID on a Franka robot).
- Status: preprint. *External source, not reproduced here.*

**S17. Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems** — Alva West, Yixuan Weng, Minjun Zhu, Zhen Lin (Westlake University), Zhiyuan Ning, Yue Zhang; 2025; https://arxiv.org/abs/2509.10401.
- Claim: explicitly scaffolding abductive causal inference improves automated failure attribution over direct LLM diagnosis on multi-agent traces.
- Status: preprint. *External source, not reproduced here.*

**Synthesis.** Counterfactual substitution is established (S9), hindsight relabeling is established (S10), and difference rewards provide the formal credit-assignment lineage (S11) — our mechanisms are not novel in kind. The sobering numbers are S12 (agent 53.5%, step 14.2%) and the caveat S15 (multiple valid attributions): a single "the Eye was to blame" label may be underdetermined. S13 and S14 say *trace representation* is a large lever, which favours our typed-message bus. S16 is the closest analogue to our per-agent replay, and flags its real cost: paired counterfactual rollouts.

---

## 3. Success/failure memory, RAG and iterative prompt/memory updates in planning

**S18. Reflexion: Language Agents with Verbal Reinforcement Learning** — Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao (Princeton, MIT); 2023; https://arxiv.org/abs/2303.11366.
- Claim: agents verbally reflect on feedback and keep reflective text in an episodic memory buffer, improving later trials without weight updates (91% pass@1 on HumanEval vs 80% for GPT-4 in that paper).
- Status: preprint (venue not stated in arXiv record). *External source, not reproduced here.*

**S19. Voyager: An Open-Ended Embodied Agent with Large Language Models** — Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar (NVIDIA, Caltech, UW); 2023; https://arxiv.org/abs/2305.16291.
- Claim: an ever-growing executable skill library plus iterative prompting that consumes environment feedback, execution errors and **self-verification** yields 3.3x more unique items and up to 15.3x faster tech-tree progress in Minecraft, with skills transferring to new worlds.
- Status: preprint. *External source, not reproduced here.*

**S20. ExpeL: LLM Agents Are Experiential Learners** — Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, Gao Huang; 2023 (AAAI 2024); https://arxiv.org/abs/2308.10144.
- Claim: an agent extracts natural-language insights from a set of training tasks and retrieves its insights and past experiences at inference, improving without parametric updates.
- Status: peer-reviewed (AAAI 2024). *External source, not reproduced here.*

**S21. EmbodiedRAG: Dynamic 3D Scene Graph Retrieval for Efficient and Scalable Robot Task Planning** — Meghan Booker, Grayson Byrd, Bethany Kemp, Aurora Schmidt, Corban Rivera (Carnegie Mellon University); 2024; https://arxiv.org/abs/2410.23968.
- Claim: retrieving task-relevant 3D scene subgraphs instead of stuffing the whole graph into the prompt cuts input tokens by an order of magnitude and average planning time by up to 70% while improving success on AI2Thor single-arm tasks, with a quadruped deployment.
- Status: preprint. *External source, not reproduced here.*

**S22. SayComply: Grounding Field Robotic Tasks in Operational Compliance through Retrieval-Based Language Models** — Muhammad Fadhil Ginting, Dong-Ki Kim, Sung-Kyun Kim, Bandi Jai Krishna, Mykel J. Kochenderfer, Shayegan Omidshafiei, Ali-akbar Agha-mohammadi; 2024; https://arxiv.org/abs/2411.11323.
- Claim: a hierarchical database of operational/environment/embodiment manuals with tree-based RAG generates plans that comply with domain rules and outperforms standard RAG in simulation and hardware.
- Status: preprint. *External source, not reproduced here.*

**S23. Retrieval-Augmented Robots via Retrieve-Reason-Act** — Izat Temiraliev, Diji Yang, Yi Zhang; 2026; https://arxiv.org/abs/2603.02688.
- Claim: actively retrieving and grounding visual procedural manuals in a Retrieve-Reason-Act loop outperforms zero-shot reasoning and few-shot example retrieval on a long-horizon assembly benchmark.
- Status: preprint. *External source, not reproduced here.*

**Synthesis.** The evidence that memory helps exists but is domain-specific (Minecraft, text games, AI2Thor, assembly), not multi-arm kitchen packing. The transferable design points are: retrieve rather than dump (S21), extract short lessons from failures (S18, S20), and pair memory with self-verification (S19). Our `docs/DATAGEN_KAGGLE.md` DG-4 ("memory iteration changes the failure mix — reported, not assumed to help") is correctly cautious.

---

## 4. Verifying plans before execution (dry run, symbolic checkers, runtime verification)

**S24. Plan Verification for LLM-Based Embodied Task Completion Agents** — Ananth Hariharan, Vardhan Dongre, Dilek Hakkani-Tür, Gokhan Tur (Microsoft); 2025; https://arxiv.org/abs/2509.02761.
- Claim: an iterative Judge-LLM/Planner-LLM loop critiques and revises action sequences, reaching up to 90% recall and 100% precision on annotated TEACh actions with 96.5% of sequences converging in at most three iterations while preserving human error-recovery patterns.
- Status: preprint. *External source, not reproduced here.*

**S25. Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification** — Keshav Ramani, Vali Tawosi, Salwa Alamir, Daniel Borrajo; 2025; https://arxiv.org/abs/2510.03469.
- Claim: converting NL plans to Kripke structures and LTL for model checking on a simplified PlanBench verification task gives GPT-5 F1 96.3% with nearly always syntactically perfect formalizations, while synthesis of *semantically* perfect formal models remains open.
- Status: peer-reviewed workshop (AgenticSE Workshop at ASE 2025). *External source, not reproduced here.*

**S26. RepV: Safety-Separable Latent Spaces for Scalable Neurosymbolic Plan Verification** — Yunhao Yang, Neel P. Bhatt, Pranay Samineni, Rohan Siva, Zhanyang Wang, Ufuk Topcu (UT Austin); 2025; https://arxiv.org/abs/2510.26935.
- Claim: a small learned projector embeds plans plus LM rationales into a space where safe/unsafe plans are linearly separable, giving a probabilistic correctness estimate and up to +15% compliance prediction accuracy with <0.2M parameters.
- Status: preprint. *External source, not reproduced here.*

**S27. Runtime Verification and Field-based Testing for ROS-based Robotic Systems** — Ricardo Caldas, Juan Antonio Pinera Garcia, Matei Schiopu, Patrizio Pelliccione, Genaina Rodrigues, Thorsten Berger; 2024; https://arxiv.org/abs/2404.11498.
- Claim: from a literature review, mined ROS repositories and questionnaires (55 answers), the paper gives 20 guidelines (8 developer, 12 QA) for architecting robots for runtime verification and field-based testing because simulation cannot realistically emulate the real world.
- Status: peer-reviewed (IEEE Transactions on Software Engineering 2024). *External source, not reproduced here.*

**S28. Digital Twin Enabled Runtime Verification for Autonomous Mobile Robots under Uncertainty** — Joakim Schack Betzer, Jalil Boudjadar, Mirgita Frasheri, Prasad Talasila; 2024; https://arxiv.org/abs/2412.09913.
- Claim: TeSSLa safety/performance monitors plus a cloud digital-twin watchdog check and can override robot actuations in real time, reducing the speed-alignment gap by up to 41% under sensor noise and environment variation.
- Status: peer-reviewed (DSRT 2024). *External source, not reproduced here.*

**Synthesis.** Two distinct layers appear: *pre-execution* verification (S24–S26, statistical/LLM-based, with measured precision/recall on datasets) and *runtime* monitoring/fail-safe (S27–S28, formal properties checked against observed behaviour). The limits are explicit: S24 is prompt-based and evaluated on human-annotated TEACh actions, S25 says semantic formalization is unsolved, S26's "guarantee" is probabilistic and learned from a model checker's labels, and S8 reports that a collision-checking verifier still leaves deployment limited by 3D waypoints. A dry run is a filter, not a proof.

---

## 5. Safe physical interaction (impedance/compliance, force limits, human push)

**S29. Safe Physical Human-Robot Interaction through Variable Impedance Control based on ISO/TS 15066** — Armin Ghanbarzadeh, Esmaeil Najafi; 2023 (journal 2024); https://arxiv.org/abs/2311.13814.
- Claim: a variable-impedance cobot controller following ISO/TS 15066 power-and-force-limiting mode achieves higher operating speeds while maintaining compliance, tested on two robot platforms.
- Status: peer-reviewed (International Journal on Interactive Design and Manufacturing 2024). *External source, not reproduced here.*

**S30. Toward Interaction Dynamics: A Predictive Framework for Safe Physical Human Robot Interaction** — Yongyan Cao, Jinshan Tang; 2026; https://arxiv.org/abs/2606.08281.
- Claim: an offset-free interaction-error MPC on a 1 kHz MuJoCo 7-DOF Franka FR3 reduces steady-state error under a repeated 15 N step from 2.73 mm to 0.033 mm at unchanged peak joint power, while a calibrated impedance baseline reaches 2.63 mm (1.36 mm with measured-force cancellation) and a physical UR5e shows 1.505 mm → 0.023 mm with the estimator, so constant-load rejection is not unique to MPC.
- Status: preprint. *External source, not reproduced here.*

**S31. Variable Stiffness Control with Strict Frequency Domain Constraints for Physical Human-Robot Interaction** — Wulin Zou, Pu Duan, Yawen Chen, Ningbo Yu, Ling Shi; 2020; https://arxiv.org/abs/2008.03663.
- Claim: gain-scheduled variable stiffness with passivity, actuator-saturation and disturbance-rejection constraints expressed per frequency band improves on a fixed-gain passivity-based PID in simulation and experiment.
- Status: peer-reviewed (IROS 2020). *External source, not reproduced here.*

**S32. Trajectory Deformations from Physical Human-Robot Interaction** — Dylan P. Losey, Marcia K. O'Malley (Rice University); 2017; https://arxiv.org/abs/1710.09871.
- Claim: combined with impedance control, force applied during pHRI can deform the robot's *future desired trajectory* without constant guidance, reducing human effort and improving movement quality in simulation.
- Status: peer-reviewed (IEEE Transactions on Robotics). *External source, not reproduced here.*

**Additional, not a retrieved source record.** `docs/IMPEDANCE_SPEC.md` §6 refers to ISO 10218-2:2025 force/pressure caps, but no standards text or vendor datasheet was retrieved in this pass (the ISO and Franka pages I attempted returned 404/403). The repo itself still labels those caps "placeholder". Nothing above is a safety certificate.

**Synthesis.** The literature supports our architecture choices (1 kHz impedance/effort loop, bounded force/torque, yield-then-return) and shows that compliance with a standard is *claimed only under tested conditions* (S29), that simpler baselines can match predictive controllers on constant loads (S30), and that passivity/frequency constraints are the usual way to make compliance stable (S31). Notice S32: yielding can also mean modifying the *plan*, which is exactly our pause-and-replan question — but S32 is simulation-only.

---

## How this changes our plan

1. **Add S4 (NSDL, `docs/findings/prior_work.md`:S4) to `paper/sections/02_related.tex` now**: it is the closest published analogue to our Brain + deterministic-solver design, and its LOCUS-Bench is public, so the "anonymized review artifact, neither cited nor compared" note at `paper/sections/02_related.tex:20` is now stale — verify and update with a citation, not a novelty claim.
2. **Resolve the other named-only TODO** at `paper/sections/02_related.tex:8` with a concrete reading list from S2, S6, S7 (CoELA, InteractGen, RoboTwin 2.0); CHORUS/D-VLC/MAAP/C3 remain unopened and should stay unclaimed.
3. **Annex the credit-assignment lineage** (S9 COMA, S10 HER, S11 Dr.Reinforce) in `paper/sections/02_related.tex` so the counterfactual Eye credit and hindsight relabeling are presented as interface-level *instantiations*, not new mechanisms.
4. **Add a sober attribution baseline** to `paper/sections/02_related.tex` and `docs/AGENT_BUILD_SPEC.md` WP-G: quote S12's 53.5% agent / 14.2% step as the state of the art on LLM-agent logs, and do not imply our fixture replay solves that problem.
5. **Capture full traces, not just completions**: extend `rtp/datagen_run.py` records to store the per-turn *inputs* (system/rules digest, memory note ids, tool call arguments and results), since S13 reports up to +76% attribution accuracy from full traces; this is a schema addition, not new infrastructure.
6. **Emit a typed per-message trace artifact** from the `docs/SCHEMAS.md` bus (`SCENE`, `PLAN`, arm events) with sender and step ids into `episodes.jsonl`, so every claim is addressable for replay and attribution (S13, S14).
7. **Add a no-op control to the oracle replay** in the attribution code path: substitute a module's output with its *identical* recorded output and require the same verdict; S12 shows some attribution methods score below random, so a null control is needed before believing any credit number.
8. **Allow multi-label responsibility** in `rtp/datagen_ingest.py` / `rtp/datagen_memory.py` (store a set of implicated modules plus confidence) — S15 argues single-root-cause labels are frequently wrong; keep the existing single-cause histogram as a separate field for continuity.
9. **Turn the success/failure memory into a small typed index** keyed by (scene family, failure cause, condition) and retrieve only a few notes in `rtp/datagen_memory.py` and `ops/kaggle/multiturn/build_task.py`; S21 reports ~10x token and up to 70% planning-time savings from retrieving rather than dumping context.
10. **Add an insight-extraction step** (one-line lessons from verified failures) when rebuilding `memory_it(N+1).json`, with the existing replay verifier as the fact-checker — the Reflexion/ExpeL/Voyager pattern (S18–S20); keep DG-4's requirement to report the failure-mix shift rather than assume improvement.
11. **Split the dry run into two logged verdicts** in `rtp/harness.py`: deterministic symbolic feasibility first, then an optional LLM Judge critique only on plans that pass it; S24's 90% recall / 100% precision is an upper reference from TEACh, not our expected number.
12. **Record verifier false accepts/rejects** on a small labelled validation slice (`reports/` run cards) and keep them out of pooled success metrics; S25 warns that syntactic formalizations can be perfect while semantic ones are not, and S26's guarantee is probabilistic.
13. **Add a message-bus runtime monitor** below the Brain — typed-schema conformance, no execution while `YIELDED`, clamp/leash checks — following the ROS field-testing guidelines (S27) and watchdog pattern (S28); it is a fail-safe layer, not a planner component.
14. **Keep compliance reporting as a cross-check row only**: use S30's 1 kHz FR3 / 15 N step shape to design `reports/g6_newton_push.json` rows, cite S29/S31 for the passivity and force-limiting vocabulary, and retain "placeholder" on force caps in `docs/IMPEDANCE_SPEC.md` §6 until ISO 10218-2:2025 is actually read.
15. **Extend the "Do not" list in `docs/AGENT_BUILD_SPEC.md` §7**: do not quote 97.0% (S4), 90/100% (S24), 367% (S7) or 3x (S16) as our numbers, and do not call any `FIXTURE`/`SIM-POSE`/`SIM-ABSTRACT` push result a hardware-safety result.

---

## What we should NOT claim

- **Not novel mechanisms.** Interface-level attribution, counterfactual substitution, hindsight relabelling and plan checking all precede us (S4, S9, S10, S11, S24–S26). We can claim a particular *integration and evidence discipline*, not the ideas.
- **No safety claim.** We make no claim that the arms are safe around people. S29's ISO/TS 15066 compliance is a claim about one tested controller under tested conditions; it does not transfer to our arms, and we have not retrieved ISO 10218-2:2025 text or a Franka datasheet. Hardware sign-off stays on the separate track (D-22).
- **No certification from placeholders.** The force/pressure/speed caps in `docs/IMPEDANCE_SPEC.md` §6 are explicitly "placeholder"; G-6 is a simulation-only cross-check and G-6b a SIM-ABSTRACT point mass. Neither is a standard-conformance test.
- **No transfer of other systems' scores.** 97.0% OOD on VIKI-L2 (S4), 90% recall/100% precision on TEACh (S24), F1 96.3% on simplified PlanBench (S25), 367% on RoboTwin (S7), 3x attribution error (S16) and 53.5%/14.2% on Who&When (S12) are their benchmarks, not ours.
- **No claim that memory/RAG helps our team.** S18–S23 are Minecraft, text games, AI2Thor and assembly; our own it0-vs-it1 evidence (DG-4) has not been run. Say "reported elsewhere; unmeasured here".
- **No claim that a verifier guarantees feasibility or safety.** S24 is prompt-based on annotated data, S25 says semantic formalization is unsolved, S26's guarantee is learned and probabilistic, and S8 shows a collision verifier coexists with deployment-limiting waypoint errors. Our dry run is a filter with measured false accepts/rejects or it is nothing.
- **No claim that attribution is uniquely determined.** S15 shows multiple plausible culprits; a single "Eye to blame" label from a fixture is a hypothesis until checked against the full traces (S13) and a no-op control.
- **No claim of superiority over NSDL/InteractGen/RoCo/CoELA/VIKI-R** (S1–S6) — no comparison matrix exists, and NSDL already occupies the "VLM sketch + deterministic solver" position.
- **No pooled evidence classes and no unreported runs.** Keep FIXTURE / SIM-POSE / SIM-ABSTRACT / SIM / MODEL / HARDWARE separate, keep frontier-model CoT and arm certification marked "not run", and generate every table from run cards as `docs/AGENT_BUILD_SPEC.md` §0 requires.
