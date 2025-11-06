# 💻 Complete LLM Prompt: Combined Sentiment Analysis Implementation

**Goal:** Implement a Python solution for a combined sentiment analysis assignment using K-Means and k-NN, focusing on token efficiency and using the provided data corpus.

### 1. Technical and Environment Requirements

1.  **Model & Efficiency:** The solution **must** use **Claude 3 Haiku** for any LLM-based task to ensure the most economical token usage.
2.  **Implementation:** The entire pipeline **must** be implemented within a **single, runnable Python script** using standard machine learning libraries (Scikit-learn, numpy) and the `dotenv` library.
3.  **API Key Management:** The script must load the Claude API Key securely from a **.env file** using the variable `CLAUDE_HAIKU_API_KEY`. (Assume the .env file exists with the key set).

### 2. Data Corpus (Simulated from The Pillars of the Earth)

The following sentences comprise the training set (30 sentences) and the test set (5 sentences), along with their manual labels (A, B, C).

#### Training Dataset (30 sentences, Equally Weighted):

| ID | Sentence | Manual Label (A, B, C) | Category Theme |
| :--- | :--- | :--- | :--- |
| 1 | The master builder drew a perfect arch. | C | Architecture/Building |
| 2 | Their greatest hope was the new cathedral. | A | Hope/Aspiration |
| 3 | A sudden blow struck his jaw, bringing blood. | B | Conflict/Violence |
| 4 | He dreamt of spires reaching to the clouds. | A | Hope/Aspiration |
| 5 | The stone was cold and perfectly cut for the wall. | C | Architecture/Building |
| 6 | Soldiers rode into the village, swords drawn. | B | Conflict/Violence |
| 7 | Aliena planned a trading route across the forest. | A | Hope/Aspiration |
| 8 | The wooden scaffolding was dangerously high. | C | Architecture/Building |
| 9 | Poverty was a curse they hoped to escape. | A | Hope/Aspiration |
| 10 | The Abbot prayed for guidance and good fortune. | A | Hope/Aspiration |
| 11 | They built the walls thick against any siege. | C | Architecture/Building |
| 12 | The fighting left many dead in the muddy field. | B | Conflict/Violence |
| 13 | He swore revenge on the man who killed his father. | B | Conflict/Violence |
| 14 | The foundation stones settled deep into the earth. | C | Architecture/Building |
| 15 | Philip wished only for peace in the priory. | A | Hope/Aspiration |
| 16 | The high windows let in light to the nave. | C | Architecture/Building |
| 17 | The Bishop's men seized all the grain stores. | B | Conflict/Violence |
| 18 | Tom yearned to see his design finally built. | A | Hope/Aspiration |
| 19 | They defended the city walls until nightfall came. | B | Conflict/Violence |
| 20 | The vaulting ribs rose gracefully toward the ceiling. | C | Architecture/Building |
| 21 | Richard planned his own redemption and future. | A | Hope/Aspiration |
| 22 | The great tower stood as a symbol of their faith. | C | Architecture/Building |
| 23 | A riot broke out near the market square. | B | Conflict/Violence |
| 24 | The young apprentice learned to carve the gargoyles. | C | Architecture/Building |
| 25 | Her determination was a formidable weapon. | A | Hope/Aspiration |
| 26 | The battlements offered a clear view of the enemy. | B | Conflict/Violence |
| 27 | They aspired to build the most beautiful church. | A | Hope/Aspiration |
| 28 | The roof beams were cut from the ancient oak. | C | Architecture/Building |
| 29 | The ambush caught them completely by surprise. | B | Conflict/Violence |
| 30 | She dreamed of regaining her family's lands. | A | Hope/Aspiration |

#### Test Set (5 sentences):

| ID | Sentence | Manual Label (Expected A, B, C) |
| :--- | :--- | :--- |
| T1 | The new market brought prosperity and trade. | A |
| T2 | His life was dedicated to the glory of God's house. | C |
| T3 | The brutal beating was a lesson for all to see. | B |
| T4 | They carefully installed the massive stained glass. | C |
| T5 | He vowed never to surrender his dignity again. | A |

### 3. Required Python Pipeline Steps

The Python script must execute the following steps sequentially:

1.  **Setup and Data Load:**
    * Load the `CLAUDE_HAIKU_API_KEY` from .env.
    * Define the **Training Dataset** and **Test Set** (sentences and manual labels A, B, C) as lists/arrays within the script.
2.  **Vectorization and Normalization:**
    * **Crucial Constraint:** Since external dependencies are discouraged for the LLM agent, **simulate the vectorization process** for all 35 sentences (Training + Test) using a **simple TfidfVectorizer** from Scikit-learn if possible, or use **dummy numpy arrays** to represent the vectors if TfidfVectorizer is problematic for the agent. *(Note: Using a simple TfidfVectorizer is preferred over complex LLM API calls for embedding, to manage tokens)*.
    * **Normalize** all generated vectors using L2 normalization.
3.  **K-Means Clustering:**
    * Run K-Means ($K=3$) on the normalized training vectors.
    * Assign the resulting cluster labels ($\alpha, \beta, \gamma$) to the training data.
    * **Report:** Calculate and print the **accuracy/alignment** between the manual labels (A, B, C) and the K-Means labels ($\alpha, \beta, \gamma$).
    * **Analyze:** Provide a brief Python `print` statement analyzing the K-Means clusters ($\alpha, \beta, \gamma$)—i.e., *what* common theme they seem to represent.
4.  **k-NN Classification:**
    * Run k-NN ($k=5$) on the Test Set (5 sentences).
    * **Prediction 1 (Using K-Means Labels):** Train k-NN using the training data labeled with $\alpha, \beta, \gamma$. Predict the labels for the Test Set.
    * **Prediction 2 (Using Manual Labels):** Train k-NN using the training data labeled with A, B, C. Predict the labels for the Test Set.
5.  **Output and Conclusion:**
    * Print a clear table or list summarizing the predictions from both k-NN runs for the 5 Test Sentences.
    * Conclude with a final `print` statement analyzing which training label set (the manual A, B, C or the K-Means $\alpha, \beta, \gamma$) resulted in a more cohesive or successful classification for the Test Set.