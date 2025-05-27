
### **Adding a True Negative Set to Each Subfamily Test Set**

**Goal:**
For each subfamily’s test set, add a *true negative* set. The size of the negative set will match the size of the test set, except when the test set contains fewer than 5 proteins—in which case, the negative set will contain 5 proteins.

#### **Rules for Constructing the True Negative Set:**

1. **Total Test Set**:
   The final test set for each subfamily will include the original test set plus its negative control set.

2. **Selecting Negatives:**

   * If a subfamily is **not assigned to any superfamily**, select negatives from other families that **are assigned to superfamilies**.
   * The selected negative proteins must not belong to the same subfamily as the current test set.
   * Negatives can be sampled from any family within another superfamily, as long as those proteins are not members of the same subfamily being tested.

3. **Minimum Negative Set Size:**

   * If the test set for a subfamily contains fewer than 5 proteins, the negative control set will contain 5 proteins.
   * Otherwise, the negative set will be the same size as the test set.

#### **Example:**

If a subfamily has 10 members, and using your splitting strategy, 8 go into the training set and 2 into the test set:

* The negative control set will have 5 proteins (since the test set is less than 5).
* These 5 negatives will be chosen from other superfamilies (excluding any from the same subfamily).
* The total test set for this family will then be 7 proteins: 2 original test set + 5 negatives.

#### **Reference:**

Definitions of subfamily and family can be found in `@input_req.md`.










Actually Used:

### How to Set the Size of the Negative Control Set

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

Do not remove any previous report files, process, or foramts. You should keep them
