### **Adding a True Negative Set to Each Subfamily Test Set**

**Goal:**
For each subfamily’s test set, add a *true negative* set. The size of the negative set will match the size of the test set, except when the test set contains fewer than 5 proteins—in which case, the negative set will contain 5 proteins.

**Selection Criteria:**

* For each subfamily, select negative control proteins from *other* superfamilies.
* Ensure that none of the negative proteins belong to the same subfamily as the target family.

**Negative Control Set Size:**

* The size of the negative control set **should match** the size of the test set for that family.
* **Exception:** If the test set contains fewer than 5 proteins, the negative control set should include **5 proteins**.

**Test Set Composition:**

* For each subfamily, the final test set = original test set (positives) + negative control set (negatives).

**Special Case:**

* If a subfamily does **not** have a superfamily assignment, the negative control proteins should be selected **only** from families that are assigned to a superfamily.

**New Evaluation Report Section**

* At the subfamily level, calculate TP, FN, TN, and FP.
* Metrics are summed across families and then averaged to obtain the overall performance.

---

**Example:**
Suppose a subfamily has 10 members. According to the splitting strategy, 8 members are used for training and 2 for testing (test set).

* For the negative control set, since the test set has fewer than 5 proteins, select 5 negative proteins from other superfamilies (excluding the same subfamily or families without a superfamily assignment).
* The total test set size for this family will therefore be 2 (positive) + 5 (negative) = 7 proteins.


### General Breakdown

For each subfamily/family, the evaluation creates a **separate binary classification problem**:

- **Positive samples**: Test proteins that actually belong to this specific subfamily/family
- **Negative samples**: Carefully selected negative control proteins from other superfamilies

### The Binary Classification Per Class

For each subfamily/family, the question becomes: *"Can the model correctly distinguish proteins that belong to this specific subfamily/family from proteins that don't belong to it?"*

### TP/TN/FP/FN Breakdown

- **TP (True Positive)**: Model correctly predicts a protein belongs to the target subfamily/family
- **FN (False Negative)**: Model incorrectly predicts a test protein as belonging to some OTHER subfamily/family (should have been the target)
- **TN (True Negative)**: Model correctly predicts a negative control protein as belonging to some OTHER subfamily/family (correctly rejects the target)
- **FP (False Positive)**: Model incorrectly predicts a negative control protein as belonging to the target subfamily/family