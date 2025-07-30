### **Adding a True Negative Set to Each Subfamily/Family Test Set**

**Goal:**
For each subfamily's or family’s test set, add a *true negative* set. The size of the negative set will match the size of the test set, except when the test set contains fewer than 5 proteins—in which case, the negative set will contain 5 proteins.

**Selection Criteria:**

*   For each subfamily or family, select negative control proteins from **within the same superfamily, but from different families**.
*   This ensures that the model is tested on its ability to distinguish between closely related families.

**Negative Control Set Size:**

*   The size of the negative control set **should match** the size of the test set for that family.
*   **Exception:** If the test set contains fewer than 5 proteins, the negative control set should include **5 proteins**.

**Test Set Composition:**

*   For each subfamily or family, the final test set = original test set (positives) + negative control set (negatives).

**Special Case:**

*   If a subfamily or family does **not** have a superfamily assignment, no negative controls will be generated for it. And those subfamily or family will be excluded from the evaluation and the test set.

**New Evaluation Report Section**

*   At the subfamily/family level, calculate TP, FN, TN, and FP.
*   Metrics are summed across families and then averaged to obtain the overall performance.

---

**Example:**
Suppose a superfamily contains four families: Family A, Family B, Family C, and Family D. We are generating a test set for a subfamily within Family A.

*   The subfamily has 10 members. According to the splitting strategy, 8 members are used for training and 2 for testing (positive samples).
*   For the negative control set, since the test set has fewer than 5 proteins, we select 5 negative proteins.
*   These 5 negative proteins will be selected from Families B, C, and D, as they belong to the same superfamily as Family A.
*   The total test set size for this subfamily will therefore be 2 (positive) + 5 (negative) = 7 proteins.

### General Breakdown

For each subfamily/family, the evaluation creates a **separate binary classification problem**:

-   **Positive samples**: Test proteins that actually belong to this specific subfamily/family.
-   **Negative samples**: Carefully selected negative control proteins from other families within the same superfamily.

### The Binary Classification Per Class

For each subfamily/family, the question becomes: *"Can the model correctly distinguish proteins that belong to this specific subfamily/family from proteins that belong to other families within the same superfamily?"*

### TP/TN/FP/FN Breakdown

-   **TP (True Positive)**: Model correctly predicts a protein belongs to the target subfamily/family.
-   **FN (False Negative)**: Model incorrectly predicts a test protein as belonging to some OTHER subfamily/family (should have been the target).
-   **TN (True Negative)**: Model correctly predicts a negative control protein as belonging to some OTHER subfamily/family (correctly rejects the target).
-   **FP (False Positive)**: Model incorrectly predicts a negative control protein as belonging to the target subfamily/family.