# GSoC 2026 Proposal: Standardizing the Publication, Preservation, and Reuse of ML-Based Unfolding Results

**Organization:** CERN-HSF (HEP Software Foundation)     
**Project:** Omnifold - Standardizing the publication, preservation, and reuse of ML-based unfolding results    
**Mentors:** Tanvi Wamorkar, Benjamin Nachman     
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
| HEPData integration | Reference implementation for submitting results to HEPData |
| Reference examples | End-to-end worked examples demonstrating the full workflow |

---

### Detailed Week-by-Week Timeline

#### Community Bonding Period (May 1 — May 24)
- Get up to speed with the OmniFold codebase and existing HEPData submission tools
- Review existing literature on FAIR data in HEP
- Discuss and finalize the scope and priorities with mentor
- Set up development environment and contribution workflow

---

#### Phase 1: Specification and Schema (May 25 — June 21)

**Week 1 (May 25 — May 31)**
- Review existing OmniFold output formats across multiple public analyses
- Document current inconsistencies and gaps (building on evaluation task findings)
- Draft v0.1 of the data specification document

**Week 2 (June 1 — June 7)**
- Finalize the metadata schema design in consultation with mentor
- Define required vs optional fields
- Write schema validation logic in Python

**Week 3 (June 8 — June 14)**
- Implement schema serialization and deserialization (YAML read/write)
- Write unit tests for schema validation
- Document the schema with examples

**Week 4 (June 15 — June 21)**
- Review and iterate on schema based on mentor feedback
- Begin drafting the HEPData integration specification
- Buffer week for any Phase 1 tasks that need more time

---

#### Phase 2: Core Python Package (June 22 — July 10)

**Week 5 (June 22 — June 28)**
- Set up the Python package structure
- Implement the weight file writer — takes OmniFold outputs and writes
  standardized HDF5 + metadata YAML
- Write tests for the writer

**Week 6 (June 29 — July 5)**
- Implement the weight file reader — loads standardized files and exposes
  a clean API for downstream use
- Implement observable computation tools (weighted histograms, covariance)
- Write tests for the reader and observable tools

**Week 7 (July 6 — July 10) — Midterm**
- Integration tests across writer → reader → observable pipeline
- Documentation for all implemented functions
- Submit midterm evaluation
- Address any mentor feedback before Phase 3

---

#### Phase 3: HEPData Integration and Examples (July 11 — August 17)

**Week 8 (July 11 — July 17)**
- Study the HEPData submission format and API
- Implement HEPData submission file generator from standardized format
- Test against the ATLAS Z+jets pseudodata used in the evaluation task

**Week 9 (July 18 — July 24)**
- Implement a reference example demonstrating the full end-to-end workflow:
  OmniFold output → standardized format → HEPData submission
- Write user-facing documentation and a quickstart guide

**Week 10 (July 25 — July 31)**
- Implement plotting tools for publication-quality figures
- Add support for systematic uncertainty visualization
- Write examples showing nominal vs systematic comparisons

**Week 11 (August 1 — August 9)**
- Implement optional model provenance storage
  (saving neural network weights alongside the unfolding results)
- Write reference example demonstrating full reproducibility workflow

**Week 12 (August 10 — August 17)**
- Final integration testing across the entire package
- Performance testing on large datasets
- Buffer for any remaining tasks

---

#### Final Week (August 17 — August 24)
- Polish documentation and write final project report
- Submit final work product
- Write blog post or summary for the HSF community

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