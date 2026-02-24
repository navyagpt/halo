# HALO Family of Models: Enabling HAIDEF to Interact with the Physical World

![HALO](assets/halo.png)

## Code Availability and Reproducibility

The training and inference code for each HALO component is available in its dedicated subdirectory:

- [HALO-RX](HALO-RX/)
- [HALO-OR](HALO-OR/)
- [HALO-Act](HALO-Act/)

All core pipelines are organized so they can be run and reproduced even if you do **not** have a robot. We designed the repository to make results reproducible from software-only workflows and provided assets, scripts, and module-level run instructions in each component folder.

For **physical execution** with HALO-Act, you should use the **SO-100 Arm**.

## Team
- Navya Gupta: Data Scientist at Kalderos 
  - Speciality: Multimodal Machine Learning, LLM distillation, Edge ML, product 
  - Technical Contributions: Product ideation, Development of Halo Rx, Halo OR 
- Gokul Puthumanaillam: PhD Candidate at UIUC 
  - Speciality: Robotics 
  - Technical Contributions: Development of Halo Act, connection of HALO components

## Problem Statement

Healthcare teams are under sustained staffing pressure, especially in perioperative care. Nursing shortages and burnout create a predictable failure mode where less time is available for direct patient care.This is not just a workforce issue; it is a workflow reliability issue. In surgery and aftercare, staffing strain contributes to delays, higher cognitive load, and increased risk of avoidable errors. The practical consequence is a worse experience for the people who matter most:
- Nurses face repeated interruptions and task-switching during medication routines.
- Surgeons / OR teams lose time to instrument friction and handoff delays. 
- Patients experience downstream effects from overloaded systems (slower transitions, less attention, more variability, canceled procedures due to short-staffing)(7).  

## Overall Solution
HALO is implemented as a coordinated family of models that share a common design principle: clinical assistance should be modular, auditable, and operationally useful. The family consists of HALO RX for medication-centric visual reasoning, HALO OR for surgical tool perception under closed-label constraints, and HALO Act for embodied, real-time physical execution. Together, these modules support end-to-end task completion where language instructions, scene understanding, candidate selection, robot control, and clinician approval can be composed into one coherent pipeline.

## HALO RX (Technical Details)
![HALO RX](assets/halorx.png)
![HALO RX Finetuning](assets/ft_rx.png)

HALO RX uses a fine-tuned MedSigLIP image encoder in a way that intentionally preserves and exploits image-text alignment rather than collapsing the task into a fixed image-only classifier. The training setup is built around paired medication examples where pill imagery is linked to medication label text through consistent identifiers. That pairing is critical because it allows the representation space to retain semantic structure between visual evidence and clinically meaningful medication strings instead of only optimizing for closed-set class boundaries.

The fine-tuning strategy focuses on alignment quality and retrieval reliability. In practice, this means the model is trained to keep visual embeddings and medication-text embeddings close for true pairs while separating non-matching pairs. By continuing to optimize an aligned embedding space, HALO RX supports context-aware label retrieval that can be constrained at inference time by the active prescription list. This prescription-constrained retrieval behavior is a key engineering decision: it reduces irrelevant candidates and shifts the model from generic recognition toward clinically grounded decision support. The net result is a perception module that remains flexible under label ambiguity while still producing structured, auditable outputs suitable for downstream action selection.

## HALO OR (Technical Details)
![HALO OR](assets/haloor.png)
![HALO OR Finetuning](assets/ft_or.png)

HALO OR uses MedSigLIP image embeddings with a dedicated trained classification head for instrument recognition. This is a deliberate technical choice rather than a shortcut. In contrast to open-ended medication retrieval, surgical instrument recognition in the targeted workflow behaves as a relatively closed-label problem with recurring objects, stable categories, and repeated viewing conditions. Under those assumptions, a lightweight classification head on top of a strong pretrained encoder is faster to train, easier to calibrate, and simpler to deploy in low-latency settings.

The model therefore leverages MedSigLIP as a high-quality feature backbone while concentrating task-specific optimization in the classification head and fine-tuning policy. This separation supports strong sample efficiency and operational robustness. In deployment terms, the classifier-head formulation is easier to profile, easier to monitor for class confusion, and easier to harden with thresholding and logging than a fully unconstrained retrieval stack. The architecture is intentionally pragmatic: it preserves the representational strength of MedSigLIP while matching the closed-label structure of OR tool selection tasks.

## HALO Act (Technical Details)
![HALO Act](assets/haloact.png)
![HALO Act Finetuning](assets/ft_act.png)

HALO Act is the embodiment layer and is action-tuned from the MedGemma family to support real-time physical task execution. Its training objective is not image classification or static captioning; it is control-conditioned behavior generation from multimodal context. The model consumes scene observations and task intent, then predicts control-relevant action outputs that can be executed at high frequency in robot control loops. This makes HALO Act fundamentally different from pure VLM inference modules: it is optimized for temporal consistency and executable behavior under sequential constraints.

The training data recipe incorporates demonstrations with varying fidelity levels so that the policy can learn both stable primitives and realistic execution variability. That diversity is important for robustness because deployment conditions are never identical to curated data captures. HALO Act is therefore framed as a policy component in a supervised assistance loop rather than an unconstrained autonomous agent. In integrated workflows, it executes repetitive physical subtasks, reports execution state, and hands back control points where clinician authorization is required. This preserves safety and traceability while still returning meaningful clinical capacity.

## Showcasing Workflow 1: Autonomous Medicine Administration
![Workflow 1](assets/kf2.png)

Demo output: [Demo 1](assets/demo1.mp4)

The first workflow composes HALO RX and HALO Act around a medication-administration scenario with approval control. A prescription is ingested and converted into structured task context. HALO Act executes the physical acquisition steps needed to capture relevant cabinet imagery. HALO RX then performs segmentation-aware medication labeling and candidate resolution in an aligned image-text space, with output constrained by clinically valid context. The pipeline then returns a supervised decision point where a clinician can approve or reject execution before physical delivery completes. This design preserves authority while removing repetitive mechanical burden from nursing time.

## Showcasing Workflow 2: Autonomous Surgery Helper
![Workflow 2](assets/kf3.png)

Demo output: [Demo 2](assets/Demo2.mp4)

The second workflow combines HALO OR, HALO Act, and MedASR in an operating-room support pattern. Voice instructions are converted into structured command intent, HALO OR resolves instrument class context from visual input, and HALO Act executes control steps for the requested assistive action. The workflow is structured to mimic natural clinician communication rather than requiring artificial command protocols. Technically, this demonstrates an integrated perception-language-action loop that can run in real time while remaining interpretable at each module boundary.

## Impact This Project Can Bring
HALO has the potential to drive outsized impact at a cost potentially as low as $300 per robot. That price point matters because it changes deployment from a small pilot luxury into something that can be rolled out in multiple hospital workflows where staffing pressure is acute. In surgical settings, where each minute carries real operational value, even modest delay reduction can create meaningful gains in schedule stability, throughput, and room utilization without requiring additional full-time hires.

Beyond direct payroll savings for care teams, HALO's bigger value is returning clinical capacity to them while working in tandem, improving workflow reliability, and reducing delays and cancellations for patients. The system is designed to remove repetitive, time-critical support tasks from nurses and perioperative teams so they can reallocate effort toward assessment, patient education, escalation, and other judgment-heavy care activities that are hard to automate and central to outcomes.

At the operations layer, the strongest impact is reliability. When routine support tasks are executed faster and more consistently, daily schedules become more predictable, handoffs become smoother, and cascading delays are less likely to propagate across the unit. That reliability translates into fewer last-minute disruptions, reduced overtime pressure, and lower cancellation risk, which improves both clinician working conditions and patient experience.

At the safety and outcomes layer, HALO supports standardized execution, traceable logs, and rapid exception escalation, which directly address known failure amplifiers such as fatigue, interruptions, and staffing shortages. We are not claiming that a robot alone causes downstream outcome improvements; the claim is that reducing avoidable burden on care teams creates a plausible and evidence-aligned pathway to safer medication processes and better post-procedure recovery support under real-world hospital constraints.






[1] C. P. Childers and M. Maggard-Gibbons, “Understanding Costs of Care in the Operating Room,” *JAMA Surgery*, vol. 153, no. 4, Art. no. e176233, 2018, doi: 10.1001/jamasurg.2017.6233.

[2] World Health Organization, “Nursing and midwifery,” *Fact sheet*, Jul. 17, 2025. [Online]. Available: [https://www.who.int/news-room/fact-sheets/detail/nursing-and-midwifery](https://www.who.int/news-room/fact-sheets/detail/nursing-and-midwifery). [Accessed: Feb. 24, 2026]. ([World Health Organization][1])

[3] P. Meredith, L. Turner, C. Saville, and P. Griffiths, “Nurse understaffing associated with adverse outcomes for surgical admissions,” *British Journal of Surgery*, vol. 111, no. 9, Art. no. znae215, Sep. 2024, doi: 10.1093/bjs/znae215. ([OUP Academic][2])

[4] World Health Organization, “Medication Without Harm,” *Initiative*. [Online]. Available: [https://www.who.int/initiatives/medication-without-harm](https://www.who.int/initiatives/medication-without-harm). [Accessed: Feb. 24, 2026]. ([World Health Organization][3])

[5] National Academy of Medicine, “CLINICIAN BURNOUT: A Crisis in Health Care,” infographic, 2020. [Online]. Available: [https://nam.edu/wp-content/uploads/2020/08/Clinician-Burnout-Infographic_FINAL_print.pdf](https://nam.edu/wp-content/uploads/2020/08/Clinician-Burnout-Infographic_FINAL_print.pdf). [Accessed: Feb. 24, 2026]. ([NAM][4])

[6] American College of Physicians, “ACP Recommends AI Tech Should Augment Physician Decision-Making, Not Replace It,” *ACP Newsroom*, Jun. 4, 2024. [Online]. Available: [https://www.acponline.org/acp-newsroom/acp-recommends-ai-tech-should-augment-physician-decision-making-not-replace-it](https://www.acponline.org/acp-newsroom/acp-recommends-ai-tech-should-augment-physician-decision-making-not-replace-it). [Accessed: Feb. 24, 2026]. ([American College of Physicians][5])

[7] J. I. Westbrook, A. Woods, M. I. Rob, W. T. Dunsmuir, and R. O. Day, “Association of interruptions with an increased risk and severity of medication administration errors,” *Archives of Internal Medicine*, vol. 170, no. 8, pp. 683–690, Apr. 2010, doi: 10.1001/archinternmed.2010.65.

[1]: https://www.who.int/news-room/fact-sheets/detail/nursing-and-midwifery?utm_source=chatgpt.com "Nursing and midwifery"
[2]: https://academic.oup.com/bjs/article/111/9/znae215/7763108?utm_source=chatgpt.com "Nurse understaffing associated with adverse outcomes for ..."
[3]: https://www.who.int/initiatives/medication-without-harm?utm_source=chatgpt.com "Medication Without Harm"
[4]: https://nam.edu/wp-content/uploads/2020/08/Clinician-Burnout-Infographic_FINAL_print.pdf?utm_source=chatgpt.com "CLINICIAN BURNOUT: A Crisis in Health Care"
[5]: https://www.acponline.org/acp-newsroom/acp-recommends-ai-tech-should-augment-physician-decision-making-not-replace-it?utm_source=chatgpt.com "ACP Recommends AI Tech Should Augment Physician ..."
