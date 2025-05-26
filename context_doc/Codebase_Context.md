# README: Synthetic Protein Domain Architecture Generation and Analysis

## Overview
This doc will explain the protein data, domain architectures for different protein superfamily. It simulates realistic protein compositions, incorporating subfamily-specific domain organizations, and motifs positioned according to defined architectural patterns. The resulting dataset supports downstream analyses of protein classifications

## Key Features
- **Subfamily-Specific Domain Architectures**:  
  Each protein is assigned to a subfamily (2.A.4.1, 2.A.4.2, or 2.A.4.3) and constructed from a characteristic set of domains defined for that subfamily.
  
- **Random Domain Addition**:  
  About 30% of the generated proteins receive additional random domains (from A–G). These extra domains respect predefined domain ranges and are placed without overlapping existing domains. A boolean flag (`Random_Domains_Added`) marks proteins that have random domains included.

- **Domain Positioning & Ranges**:  
  Each domain type (A–G) has a predefined positional range (e.g., Domain A might appear near the start of the protein, within 10–30 amino acids). Both core subfamily domains and added random domains are placed according to these ranges and do not overlap.

- **Domain Scoring**:  
  Each domain is assigned a confidence score. Domains expected for the subfamily have very low (negative) scores, indicating high confidence. Randomly added domains have more variable scores, simulating lower confidence in prediction.

  Motif lengths and positions are adjusted to ensure motifs do not exceed domain boundaries. If a motif does not fit, it is skipped.

- **Gaps (Separators) Between Domains**:  
  Separators are the gap regions (separators) between domains to simulate realistic spacing. These separators are stored and available for feature extraction and downstream visualization.

- **Feature Extraction**:  
  After generating the dataset, multiple feature extraction functions create additional DataFrame columns, providing detailed representations of:
  - Domain occurrences with rank, length, and score.
  - Motif occurrences with rank, length, and score.
  - Domain-to-domain separators with length and rank information.
  
  This structured output is suitable for further analysis and visualization.

## Core Functions

1. **`generate_domain(domains)`**  
   Given a list of domain types, assigns start and end positions for each domain within their predefined positional ranges. Ensures no overlap and returns a list of `(domain_type, start, end)` tuples.

2. **`generate_random_domain(row, num_new_domains=2)`**  
   Attempts to insert additional random domains into an existing protein’s domain structure. The added domains:
   - Respect the predefined domain ranges for their type.
   - Are placed without overlap.
   If it cannot find a suitable position, the attempt for that domain is skipped.

3. **`assign_scores(domains, expected_domains)`**  
   Assigns confidence scores to domains. Expected domains receive very low (negative) scores, indicating high confidence, while non-expected domains get a range of higher scores, simulating lower confidence.

5. **`calculate_spaces(domains, use_threshold=False, threshold=0)`**  
   Computes separators (gaps) between adjacent domains, returning their positions and lengths. Supports filtering by a minimum gap threshold if needed.

6. **`extract_domain_features(df)`**  
   Extracts domain-level features from the DataFrame, such as rank (order in the protein), length, and score. Each domain type gets its own column for easy filtering and analysis.

7. **`extract_motif_features(df)`**  
   Extracts motif-level features for each motif type. For each motif type, a column is created with tuples describing each motif found in the protein.

8. **`extract_separator_features(df)`**  
   Extracts gap (separator) features and places them in the DataFrame. Columns are generated for each possible domain-to-domain combination, making it easy to analyze spacing patterns.

9. **`generate_dataset(num_proteins_per_subfamily, include_motifs=False)`**  
   The main driver function that:
   - Iterates through each subfamily.
   - Generates the required number of synthetic proteins.
   - Assigns domains, optionally adds random domains, calculates protein length, assigns scores, and computes separators.
   - If `include_motifs` is True, also generates motifs for each domain.
   - Aggregates all results into a DataFrame.
   - Runs feature extraction to produce a rich, structured dataset ready for analysis.

## Usage Notes
- The code is flexible and can be extended. Patterns for motif placements can be changed as needed.
- By default, the dataset simulates realistic protein architectures from the 2.A.4 family, making it useful for testing pipelines that analyze protein domains and motifs.
- The addition of a boolean `Random_Domains_Added` flag helps distinguish proteins that follow canonical subfamily arrangements from those enriched with random domains.
  