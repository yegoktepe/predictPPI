# predictPPI

This study introduces MFPIC (Multi-Feature Protein Interaction Classifier), a novel computational model for predicting protein-protein interactions (PPIs). The model integrates enhanced sequence-based features, including Spaced Conjoint Triad (SCT) and Amino Acid Pairwise Distance (AAPD), along with established methods such as Position-Specific Scoring Matrices (PSSM) and AAindex features. MFPIC captures complex sequence motifs and spatial relationships within proteins, improving the accuracy of PPI predictions. Evaluated on Saccharomyces cerevisiae, Helicobacter pylori, and Human datasets, the model outperforms state-of-the-art methods, demonstrating its potential for biological and therapeutic research​.

Data; is the folder where the database files are located. The AAindex file is a database containing amino acid physicochemical properties, substitution matrices and statistical protein contact potentials. It can be accessed as an open source from the https://www.genome.jp/aaindex/ website.

The seqs file lists protein sequences specific to the biological organism.

The pairs file contains interaction data specific to the biological organism.
