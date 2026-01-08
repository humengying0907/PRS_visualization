# PRS Visualization Server

This project aims to build a server that lets people visualize Polygenic Risk Score (PRS) results reported by different research groups for a given trait. The goal is to provide a clear, consistent way to compare PRS score distributions, metadata, and reporting context across cohorts and methodologies.

## Project Goals

- **Centralized PRS visualization**: Offer a single place to explore PRS scores for multiple traits.
- **Cross-group comparison**: Enable side-by-side comparison of PRS results from different reporting groups.
- **Trait-focused views**: Provide dashboards for each trait that surface the most relevant PRS summaries and distributions.
- **Transparent provenance**: Clearly display which group reported each PRS and any related metadata.

## Core Tasks

1. **Ingest PRS results**
   - Define a schema for PRS score data and metadata.
   - Support uploads or API ingestion from multiple groups.

2. **Normalize and validate data**
   - Harmonize score formats and trait identifiers.
   - Validate required fields and flag inconsistencies.

3. **Store and query data**
   - Persist PRS data in a queryable database.
   - Provide endpoints for trait- and group-specific queries.

4. **Visualize PRS scores**
   - Build trait dashboards with distributions, summary stats, and comparisons.
   - Allow filtering by group, cohort, or reporting method.

5. **Serve a public interface**
   - Expose a web UI for interactive exploration.
   - Provide a documented API for programmatic access.

## Expected Outcomes

- A functional server that hosts PRS visualization dashboards for multiple traits.
- Consistent, comparable views of PRS scores reported by different groups.
- A foundation for expanding to additional traits and reporting sources.
