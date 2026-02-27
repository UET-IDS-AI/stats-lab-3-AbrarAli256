"""
Prob and Stats Lab – Discrete Probability Distributions

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 where required.
"""

import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    """
    STEP 1: Consider a standard 52-card deck.
            Assume 4 Aces.

    STEP 2: Compute analytically:
            - P(A)
            - P(B)
            - P(B | A)
            - P(A ∩ B)

    STEP 3: Check independence:
            P(A ∩ B) ?= P(A)P(B)

    STEP 4: Simulate 200,000 experiments
            WITHOUT replacement.
            Use random_state=42.

            Estimate:
            - empirical P(A)
            - empirical P(B | A)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(B | A)
            empirical P(B | A)

    RETURN:
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    """
    P_A         = 4 / 52       
    P_B         = 4 / 52       
    P_B_given_A = 3 / 51       
    P_AB        = P_A * P_B_given_A   
    rng = np.random.default_rng(42)


    n_simulations = 200_000
    deck = np.arange(52)
    draws = np.array([rng.choice(deck, size=2, replace=False) for _ in range(n_simulations)])

    first_card  = draws[:, 0] 
    second_card = draws[:, 1]  

    event_A = first_card < 4
    event_B = second_card < 4
    empirical_P_A = np.mean(event_A)
    A_occurred = event_A
    empirical_P_B_given_A = np.sum(event_A & event_B) / np.sum(A_occurred)
    absolute_error = abs(P_B_given_A - empirical_P_B_given_A)

    return (
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    )



# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    """
    STEP 1: Define Bernoulli(p) PMF:
            p_X(x) = p^x (1-p)^(1-x)

    STEP 2: Compute theoretical:
            - P(X = 1)
            - P(X = 0)

    STEP 3: Simulate 100,000 bulbs
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X = 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X = 1)
            empirical P(X = 1)

    RETURN:
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    """
    theoretical_P_X_1 = p          
    theoretical_P_X_0 = 1 - p      
    rng = np.random.default_rng(42)
    bulbs = rng.binomial(n=1, p=p, size=100_000)
    empirical_P_X_1 = np.mean(bulbs)

    absolute_error = abs(theoretical_P_X_1 - empirical_P_X_1)

    return (
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    )

# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    """
    STEP 1: Define Binomial(n,p) PMF:
            P(X=k) = C(n,k)p^k(1-p)^(n-k)

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 2)
            - P(X ≥ 1)

    STEP 3: Simulate 100,000 inspections
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 1)
            empirical P(X ≥ 1)

    RETURN:
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    """
    theoretical_P_0 = math.comb(n, 0) * (p**0) * ((1-p)**n)
    theoretical_P_2 = math.comb(n, 2) * (p**2) * ((1-p)**(n-2))
    theoretical_P_ge_1 = 1 - theoretical_P_0
    rng = np.random.default_rng(42)
    inspections = rng.binomial(n=n, p=p, size=100_000)
    empirical_P_ge_1 = np.mean(inspections >= 1)
    absolute_error = abs(theoretical_P_ge_1 - empirical_P_ge_1)

    return (
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    )
   

# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    """
    STEP 1: Let p = 1/6.

    STEP 2: Define Geometric PMF:
            P(X=k) = (5/6)^(k-1)*(1/6)

    STEP 3: Compute theoretical:
            - P(X = 1)
            - P(X = 3)
            - P(X > 4)

    STEP 4: Simulate 200,000 experiments
            using random_state=42.

    STEP 5: Compute empirical:
            - empirical P(X > 4)

    STEP 6: Compute absolute error BETWEEN:
            theoretical P(X > 4)
            empirical P(X > 4)

    RETURN:
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    """
    p = 1/6
    q = 5/6  


    theoretical_P_1 = (q**0) * p        

    theoretical_P_3 = (q**2) * p          
    theoretical_P_gt_4 = q**4             
    rng = np.random.default_rng(42)

    rolls = rng.geometric(p=p, size=200_000)

    empirical_P_gt_4 = np.mean(rolls > 4)

    absolute_error = abs(theoretical_P_gt_4 - empirical_P_gt_4)

    return (
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    )



# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    """
    STEP 1: Define Poisson PMF:
            P(X=k) = e^(-λ) λ^k / k!

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 15)
            - P(X ≥ 18)

    STEP 3: Simulate 100,000 hours
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 18)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 18)
            empirical P(X ≥ 18)

    RETURN:
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    """
    theoretical_P_0 = (math.e**(-lam) * lam**0) / math.factorial(0)
    theoretical_P_15 = (math.e**(-lam) * lam**15) / math.factorial(15)

    theoretical_P_ge_18 = 1 - sum(
        (math.e**(-lam) * lam**k) / math.factorial(k)
        for k in range(18)       
    )


    rng = np.random.default_rng(42)

    hours = rng.poisson(lam=lam, size=100_000)

    empirical_P_ge_18 = np.mean(hours >= 18)

    absolute_error = abs(theoretical_P_ge_18 - empirical_P_ge_18)

    return (
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    )

  
