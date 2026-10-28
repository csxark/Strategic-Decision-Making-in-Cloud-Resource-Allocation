# Strategic Decision-Making in Cloud Resource Allocation

A game-theoretic simulation and interactive dashboard modeling resource contention, proportional sharing, and strategic decision-making between competing cloud workloads under finite CPU capacity.

Built for a college Game Theory course demonstration using Python and Streamlit.

---

## 1. Quick Start & Launch Command

### Prerequisites
- Python 3.9+
- pip

### Installation
Install the minimal dependencies:
```bash
pip install -r requirements.txt
```

### Run the Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 2. Problem Overview & Game Theory Framing

In multi-tenant cloud environments (e.g., Kubernetes clusters, public cloud VMs), multiple applications compete for limited shared CPU capacity. 
When total resource demand exceeds available capacity, resource starvation and contention occur.

### Players and Strategies
- **Players:** Two strategic applications, **Application A** and **Application B**.
- **Strategy Space:** Each player chooses between two operational profiles:
  - **Low Demand:** Modest resource reservation ensuring cluster stability.
  - **High Demand:** Aggressive resource reservation attempting to maximize private performance.

### The Strategic Conflict (Tragedy of the Commons)
When both applications request **Low Demand**, total cluster load remains well within physical capacity. Both achieve predictable latency and high collective utility. 
However, each individual application faces an incentive to unilaterally defect to **High Demand** to secure a larger fraction of resources under fair-share throttling. 

When both defect, total demand overwhelms the cluster ($D_A + D_B > C$), triggering heavy throttling and queueing contention. This traps the system in a **Prisoner's Dilemma / Tragedy of the Commons** Nash Equilibrium with diminished social welfare.

---

## 3. Mathematical Model & Formulation

### A. Resource Allocation Policy (Proportional Fair Sharing)
Let $C > 0$ be the total available CPU capacity, and $D_i \ge 0$ be the requested demand of application $i \in \{A, B\}$.
Total cluster demand is $D_{\text{total}} = D_A + D_B$.

The allocated CPU $x_i$ to application $i$ satisfies:
$$
x_i = \begin{cases}
D_i, & \text{if } D_{\text{total}} \le C \quad (\text{Unconstrained}) \\
C \cdot \dfrac{D_i}{D_{\text{total}}}, & \text{if } D_{\text{total}} > C \quad (\text{Proportional Fair Sharing})
\end{cases}
$$

**Guarantees:**
- Capacity constraint is strictly preserved: $x_A + x_B \le C$.
- When throttled, shares are proportional to requested demands, modeling standard weighted fair queueing (WFQ) and completely fair scheduler (CFS) behaviors.

### B. Payoff Function (Parametric Formulation)
The utility $U_i$ of application $i$ is calculated from first principles:
$$
U_i = \underbrace{v \cdot x_i}_{\text{Performance Benefit}} - \underbrace{c \cdot D_i}_{\text{Reservation Cost}} - \underbrace{\delta \cdot \max(0, D_i - x_i)}_{\text{Throttling Shortfall Penalty}} - \underbrace{\gamma \cdot \max(0, D_{\text{total}} - C)}_{\text{Cluster Congestion Penalty}}
$$

Where:
- $v > 0$: Utility gain per unit of allocated CPU.
- $c \ge 0$: Unit cost of requesting/reserving resources.
- $\delta \ge 0$: Penalty per unit of unfulfilled demand due to CPU throttling.
- $\gamma \ge 0$: Cluster-wide contention penalty resulting from excessive queueing delay and context switching.

*Note: Utilities represent abstract preference values, not physical compute units.*

### C. Pure-Strategy Nash Equilibrium Computation
For each player $i \in \{A, B\}$ and opponent strategy $s_{-i}$:
1. Compute the best response set:
   $$
   BR_i(s_{-i}) = \arg\max_{s_i \in \{\text{Low}, \text{High}\}} U_i(s_i, s_{-i})
   $$
   *(Ties are fully handled: if payoffs are equal within numerical tolerance, both actions belong to the best-response set).*
2. A strategy profile $(s_A^*, s_B^*)$ is a **Pure-Strategy Nash Equilibrium (PSNE)** if and only if:
   $$
   s_A^* \in BR_A(s_B^*) \quad \text{and} \quad s_B^* \in BR_B(s_A^*)
   $$
3. **Social Welfare (Utilitarian):**
   $$
   W(s_A, s_B) = U_A(s_A, s_B) + U_B(s_A, s_B)
   $$
   The dashboard compares the Nash equilibrium welfare against the Pareto/social optimum to evaluate the **Price of Anarchy (PoA)**.

---

## 4. Design & Color Palette Compliance

This application adheres strictly to the designated palette schema:
- `--deep-space-blue`: `#003049` (Primary brand color, headings, allocated CPU bars, selected states)
- `--deep-space-blue-light`: `#00679F` (Secondary interactive elements, requested demand bars)
- `--deep-space-blue-bright`: `#00A0F7` (Nash equilibrium highlights and interactive emphasis)
- `--flag-red`: `#D62828` (Cluster capacity threshold line, error states, and throttling alerts)
- `--vivid-tangerine`: `#F77F00` (Occasional emphasis for resource utilization and key metrics)
- `--sunflower-gold`: `#FCBF49` (Secondary highlights and warning indicators)
- `--vanilla-custard`: `#EAE2B7` (Soft neutral surfaces, matrix headers, and subtle dividers)

No extraneous gradients, unapproved accents, or arbitrary colors are introduced. The design follows a minimalist, modern SaaS analytics aesthetic optimized for laptop screens.

---

## 5. Project Structure

```text
GameTheory_project/
├── app.py                 # Streamlit interface and results dashboard
├── game_theory.py         # Allocation algorithms, payoffs, best responses, and PSNE solver
├── test_game_theory.py    # Automated unit tests verifying mathematical correctness
├── requirements.txt       # Minimal Python dependencies
└── README.md              # Project documentation and theoretical foundation
```

---

## 6. Verification and Testing

Execute the unit test suite to verify:
- Proportional sharing under capacity constraints.
- Edge cases ($C=0$, zero demand, asymmetric requests).
- Best response tie handling.
- Canonical Prisoner's Dilemma, Coordination Game, and cyclic games.

```bash
python -m unittest test_game_theory.py
```
All 12 unit tests pass deterministically in $<0.01$ seconds.
