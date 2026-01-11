# PRS Locus Viewer 


> **🏆 CMU x NVIDIA 2026 Hackathon Project**
>
> This tool was developed during the **[CMU x NVIDIA 2026 Hackathon](https://guides.library.cmu.edu/hackathon/2026)** - *Federated Learning for Biomedical Applications* (January 7–9, 2026).

## Hackathon Context

**Team:** Polygenic Risk Aggregation in Common Diseases and Phenotypes  

**Goals:**  
- Build **privacy-preserving pipelines** for computing and aggregating Polygenic Risk Scores (PRS)  
- Enable **deterministic risk score aggregation** across distributed genetic datasets  
- Explore **representation learning** approaches for cross-cohort PRS integration 

🔗 **Full Team Repository:** https://github.com/collaborativebioinformatics/PRSAggretator

---

## About PRS Locus Viewer
**PRS Locus Viewer** is an interactive Dash application designed for visualizing Polygenic Risk Score (PRS) variants in their genomic context. 

It addresses the need for interpretability by allowing researchers to inspect Polygenic Risk Score variants within their full genomic context. Users can:
* **Visualize SNP effect sizes** across multiple scoring files simultaneously.
* **Map variants** to nearby genes for biological context.
* **Dynamically explore** specific loci to validate aggregation results.

## Key Features
* **Interactive Locus Zoom:** detailed inspection of specific genomic regions.
* **Cross-Score Comparison:** overlay weights from different PRS models or cohorts.
* **Gene Track Integration:** automatic mapping of variants to gene annotations.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/humengying0907/prs-locus-viewer.git
cd prs-locus-viewer
```

### 2. Create the Environment

You can set up the required dependencies using Conda. Create a file named `environment.yml` with the content below, or install manually.

**Option A: Using Conda (Recommended)**

```bash
conda env create -f environment.yml
conda activate prs_viewer
```

**Option B: Using pip**

```bash
pip install pandas dash requests pyranges
```

---
## Data Preparation

This tool runs locally and requires **Harmonized Scoring Files** (containing `hm_chr` and `hm_pos` columns) as input. A future web-based version is planned to eliminate the need for manual file uploads.

### 1. Downloading Harmonized Scores

You can download harmonized files directly from the [PGS Catalog FTP site](https://www.pgscatalog.org/):


- **Navigate to:** `PGS_ID → ScoringFiles → Harmonized`
- **Example URL:**  
  https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003402/ScoringFiles/Harmonized/
- **Example file:**
  PGS003402_hmPOS_GRCh37.txt.gz

Ensure the downloaded files include at least the following columns:

- `hm_chr`
- `hm_pos`
- `effect_weight`
- `rsID`

### 2. Using Example Data

For quick testing, sample harmonized scoring files are provided in the `example_input/` directory of this repository.

---

## Usage

To launch the viewer, run the main script with your input PRS text files and the target chromosome.

```bash
python prs_locus_viewer.py [input_file_1] [input_file_2] ... --chr [chromosome]
```

- `input_file`: Path to PRS scoring files (must contain columns: `hm_chr`, `hm_pos`, `effect_weight`, `rsID`).
- `--chr`: The specific chromosome to load (e.g., `chr5`).

### Example Run
```bash
python prs_locus_viewer.py example_input/PGS001818_hmPOS_GRCh37.txt example_input/PGS000020_hmPOS_GRCh37.txt example_input/PGS003402_hmPOS_GRCh37.txt --chr chr5
```

---

## Startup Output

Upon running the command, you should see output like:

```text
============================================================
  PRS LOCUS VIEWER | Chromosome: chr5
============================================================

[1/3] Loading data for chr5...
[2/3] Calculating gene coverage (Window: ±25kb)...
      Targeting 2,838 unique SNPs

[info] Calculating gene coverage (±25kb) for 2838 unique SNPs...
[==============================] 100% (processing chr5)
[info] Gene coverage calculation finished.
[3/3] Calculation complete.

------------------------------------------------------------
 [READY] Viewer active at: http://127.0.0.1:8050/
         (Press CTRL+C to stop)
------------------------------------------------------------

Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'prs_locus_viewer'
 * Debug mode: on
```

Once the server is ready, open your web browser and navigate to `http://127.0.0.1:8050/` to start using the tool.

## Interactive Preview

https://github.com/user-attachments/assets/9d0be08c-5592-4b8e-9200-498b36715900
