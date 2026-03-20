# GSoC 2026 Proposal: Standardizing the Publication, Preservation, and Reuse of ML-Based Unfolding Results

**Organization:** CERN-HSF (HEP Software Foundation)     
**Project:** Omnifold - Standardizing the publication, preservation, and reuse of ML-based unfolding results    
**Mentors:** Tanvi Wamorkar, Benjamin Nachman  
**Project duration:** 175 hours    
**Contributor:** Omkar Anand Mishra     
**GitHub:** [https://github.com/Nik172](https://github.com/Nik172)    
**LinkedIn:** [Link](https://www.linkedin.com/in/omkar-anand-mishra-675377321/)    
**Email:** omkar172ftw@gmail.com     

---

## Synopsis

In this project, we aim to create a Python package that produces OmniFold weights in a standardized format, accompanied by data and metadata specifications for
per-event weights, dataset selections, and optional model information. The project will also provide user-facing tools to apply weights, compute observables, and generate 
publication-quality plots, along with reference examples demonstrating integration with HEPData. This will enable physicists to easily publish, preserve, and reuse ML-based 
unfolding results, advancing the FAIR data ecosystem in HEP and maximizing the impact of these analyses.

---

## About Me

I am a second year B.Tech student at the Indian Institute of Technology(IIT) Bhubaneswar. I have experience working in machine learning, data science, and NLM. I am deeply fascinated by particle physics and enjoy reading about the latest developments in the field. I am interested in this project because it gives me the opportunity to contribute to the physics community and use my software and AIML skills to make something meaningful that will be helpful for physicists around the world.

### Relevant Skills
- **Programming:** Python, C, C++, bash
- **Libraries:** NumPy, Pandas, Matplotlib, PyTorch, Scikit-learn,
  h5py, pytest, Hugging Face Transformers, LangChain
- **Tools:** Git, GitHub, Jupyter, Linux
- **Domain Knowledge:** Machine learning, scientific data formats, software testing, neural network training, data pipelines, and a strong interest in high energy physics

### My Prior Works
- **GSoC 2026 Evaluation Task** - analyzed three OmniFold HDF5 files from an
  ATLAS Z+jets pseudodata dataset, designed a 5-layer YAML metadata schema,
  and implemented a tested weighted histogram function with ensemble uncertainty
  bands.
  [Repository](https://github.com/Nik172/OmnifoldUnfoldingAndHistogram)

- **Causal Retrieval System for Conversational Analytics** - hybrid GraphRAG +
  Vector RAG system built for Inter-IIT Tech Meet '25. Designed the preprocessing, 
  embedding, and storage pipelines for a dataset of about 20,000 real conversation transcripts.
  Achieved 0.74 faithfulness and 0.77 relevance scores.
  [Repository](https://github.com/Nik172/ByteMe-Causal-extractor)

- **PDD Dysarthria Speech Normalizer** - fine-tuned OpenAI Whisper with LoRA
  adapters on dysarthric speech corpus, reducing Word Error Rate by ~18%.
  Built an end-to-end ASR evaluation pipeline with custom audio preprocessing.
  [Repository](https://github.com/Nik172/EazeVoice)

- **Next Word Predictor using Transformer** - fine-tuned GPT-2 on WikiText-2
  achieving a mean test perplexity of 17.36 using HuggingFace Trainer with
  gradient accumulation and optimized batching.
  [Repository](https://github.com/Nik172/ML_next_word_predictor)

- **LLM Powered Autonomous Research Agent** - LangChain RAG pipeline with
  Groq LLaMA-3, FAISS vector store, and research paper QA functionality.
  [Repository](https://github.com/Nik172/AI_RAG_Research_Agent)

- **STAGATE Implementation and Perturbation Analysis** - implementation and
  perturbation analysis of the STAGATE spatial transcriptomics model.
  [Repository](https://github.com/Nik172/Basic_STAGATE_implementationAndPerturbationAnalysis)

---

## Background and Motivation

### The Problem
Modern unfolding methods like OmniFold produce per-event weights rather than
fixed binned histograms. This enables flexible reinterpretation of results and
a physicist can apply published weights to compute any observable they choose,
long after the original analysis was published.

However, there is currently no standard for how these weights should be stored,
documented, or shared. As part of the evaluation task for this project, I
analyzed three OmniFold HDF5 output files from an ATLAS Z+jets pseudodata
measurement and found:

- **Zero metadata** present in any of the three files despite 200 columns
  across 418,014 events
- **No documentation** of event selection criteria, normalization, or generator
  provenance
- **Structurally inconsistent files** - the three related files had 200, 51,
  and 26 columns respectively with no schema governing their relationship

### Why This Matters
This project directly addresses the HEP community's broader push toward FAIR
data - Findable, Accessible, Interoperable, and Reusable. Currently due to the
lack of standardization, physicists who want to reuse OmniFold results must
reverse-engineer the files or contact the original authors for clarification,
or risk misinterpreting the data. This creates a barrier to reuse and limits
the impact of these unfolding results.

Defining a specification and tooling for OmniFold results brings ML-based
unfolding into the same open science ecosystem that HEPData, REANA, and
RECAST are building across the rest of HEP.

---

## Project Plan

### Overview of Deliverables

| Deliverable | Description |
|---|---|
| Data/metadata specification | Formal schema defining the standard format for OmniFold weights and metadata |
| Python package | Tools to produce and consume standardized OmniFold outputs |
| Reference examples | End-to-end worked examples demonstrating the full workflow |
| HEPData integration(stretch goal) | Reference implementation for submitting results to HEPData |

---

### Detailed Week-by-Week Timeline

#### Community Bonding Period (May 1 - May 24)
- Get up to speed with the OmniFold codebase and existing HEPData submission tools
- Review existing Omnifold output formats across multiple public analyses to understand current practices
- Review existing analysis preservation tools (REANA, RECAST)
- Discuss with the mentors and finalize what are the required vs optional fields for the metadata schema design
- Discuss and finalize the scope and priorities for the project with the mentors

---

#### Phase 1: Specification and Schema (May 25 - June 15)

**Week 1 (May 25 - May 31) - data and metadata specification**
- Survey existing OmniFold outputs across multiple public analyses to document current inconsistencies (building on evaluation task findings)
- Draft v0.1 of the formal metadata schema (YAML) covering: dataset identity, simulation provenance, event selection, normalization, and cross-section information
- Define required vs optional fields with justification for each decision

**Week 2 (June 1 - June 7) - Per-Event Weight Format**
- Discuss with mentors and finalize the metadata schema
- Define the standard container format for per-event weights (HDF5 or Parquet)
- Specify naming conventions and indexing for: nominal weights, statistical replicas, bootstrap variations, and systematic variations
- Implement and test schema validation logic in Python
- Ensure the format scales to large datasets

**Week 3 (June 8 - June 14) - Model & Training Details**
- Define minimal vs full model publication options
- Standardize storage format for: trained model checkpoints, architecture definitions, training hyperparameters, and feature preprocessing details
- Clearly distinguish which components are required for reinterpretation vs full retraining
- Write unit tests for schema serialization and deserialization

---

#### Phase 2: Core Python Package (June 22 - July 10)

**Week 4 (June 15 - June 21) - Observable Definitions**
- Define a standard observable description format including:
  physics definitions, phase space restrictions, object definitions, and binning conventions
- Implement reference Python implementations for common observables
- Write tests verifying observable computation against known results

**Week 5 (June 22 - June 28) - Package Structure and Weight I/O**
- Set up the Python package structure with separate modules for:
  reading, writing, validation, observables, plotting, and HEPData export
- Implement a "weight file writer" that takes OmniFold outputs and produces standardized HDF5 + metadata YAML
- Implement a "weight file reader" that loads standardized files and exposes a clean API for downstream use
- Write integration tests for the full writer to reader pipeline

**Week 6 (June 29 - July 5) - Analysis and Reinterpretation API**
- Implement the user-facing Python API for:
  loading published weights (locally or from HEPData), applying weights to events, computing arbitrary observables, and propagating statistical and systematic uncertainties
- Build plotting tools supporting common analysis workflows and publication-quality figures
- Write examples demonstrating the API on the ATLAS Z+jets pseudodata

**Week 7 (July 6 - July 10) - Midterm**
- Full integration tests across the entire pipeline:
  OmniFold output -> standardized format -> observable computation -> plots
- Documentation for all implemented modules
- Submit the total work done till midterm

---

#### Phase 3: Validation, HEPData and Examples (July 11 - August 17)

**Week 8 (July 11 - July 17) - Validation Framework**
- Implement standardized validation procedures:
  closure tests, normalization checks, and stability checks across iterations
- Define clear pass/fail criteria and expected outputs for each validation step
- Write automated validation tests that run as part of the CI pipeline

**Week 9 (July 18 - July 24) - Reference Example 1**
- Build a complete end-to-end reference example demonstrating:
  running OmniFold -> writing standardized output -> uploading to HEPData
- Write a user-facing quickstart guide and documentation
- Ensure the example is fully reproducible from a clean environment

**Week 10 (July 25 - July 31) - Reference Example 2**
- Build a second reference example demonstrating the reinterpretation workflow:
  loading published weights from HEPData -> applying to new events -> computing new observables post-publication -> generating publication plots
- Demonstrate nominal vs systematic uncertainty comparisons

**Week 11 (August 1 - August 9) - HEPData Integration Layer**
- Study the HEPData submission format and API in detail
- Implement the HEPData submission file generator from the standardized format:
  mapping OmniFold results to HEPData records, submission templates, metadata conventions, and validation steps
- As this is a stretch goal, it will be implemented only if the past work is completed on schedule with no issues.

**Week 12 (August 10 - August 17) - Buffer period and Final Polish**
- Final integration testing across the entire package
- Performance testing on large datasets
- Fix any remaining issues from mentor feedback
- Polish all documentation and API references
- Buffer for any tasks that ran over schedule

---

#### Final Week (August 17 - August 24)
- Submit the final work 
- Document future work and known limitations

---

## Technical Approach

### Metadata Schema
[Briefly describe your schema design — reference the 5-layer structure
from your evaluation task. Mention YAML format, required vs optional fields,
the tiered approach for reuse vs full reproducibility.]

### Python Package Architecture
[Describe how you'd structure the package — e.g. separate modules for
reading, writing, validation, observables, plotting, HEPData export.
Mention your approach to testing.]

### HEPData Integration
[Describe what you know about HEPData's submission format and how
the standardized files would map onto it.]

### Key Technical Challenges
- **Variable ensemble sizes** — the schema must accommodate analyses
  with different numbers of ensemble members
- **Cross-experiment naming conventions** — need a controlled vocabulary
  or mapping layer for observable names
- **Long-term reproducibility** — storing ML model weights alongside
  unfolding results without bloating file sizes
- [Add any others you can think of]

---

## Evaluation Task Summary

As part of the application process I completed all three evaluation tasks:

**Task 1 — Exploratory Data Analysis**
Analyzed three OmniFold HDF5 files from an ATLAS Z+jets pseudodata dataset.
Classified all 200 columns across 418,014 events into observables, weight
subcategories, and metadata. Identified 8 critical gaps for reuse and 8
standardization challenges across experiments.

**Task 2 — Metadata Schema Design**
Designed a 5-layer YAML metadata schema covering dataset identity, simulation
provenance, event selection, full column documentation, and OmniFold training
provenance. Wrote a justification document explaining design decisions including
the tiered required/optional structure and single-file approach.

**Task 3 — Coding Exercise**
Implemented a self-contained `weighted_histogram()` function with ensemble
uncertainty bands, robust edge case handling, and 14 passing pytest tests.

The evaluation task repository is available at:
[https://github.com/Nik172/OmnifoldUnfoldingAndHistogram](https://github.com/Nik172/OmnifoldUnfoldingAndHistogram)

---

## Availability

- **Hours per week:** [e.g. 35 hours]
- **Timezone:** IST (UTC+5:30)
- **Potential conflicts:** [Any exams, travel, internships during May-August]
- **Other GSoC applications:** [mention or not — your choice]

---

## Why Me

[3-4 sentences. Connect your specific skills and the evaluation task work
to why you're the right person for this project. Be direct and confident.]

---

## References

- OmniFold repository: https://github.com/wamorkart/omnifold-hepdata
- OmniFold paper: https://arxiv.org/abs/1911.09107
- HEPData: https://www.hepdata.net
- FAIR data principles: https://www.go-fair.org/fair-principles/