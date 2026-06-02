---
title: "GMM and EM"
date: "none"
tags: ["machine-learning"]
---

Consider this hero level statistics from an RPG game: instead of an expected bell-shaped distribution, we observe a 
bimodal distribution with high counts at both ends and a dip in the middle.

![](rpg_heros.png)

This pattern tells a compelling story about player psychology. Players tend to categorize heroes into two groups: meta 
heroes that they prioritize leveling up to maximum potential, and weak heroes that receive minimal investment. Each 
group follows its own normal distribution, and when combined, these two distributions create the bimodal pattern we 
observe.

This is precisely where Gaussian Mixture Models (GMM) come into play, a powerful statistical tool that allows us to 
model complex data as a combination of multiple Gaussian distributions, revealing hidden structures and patterns that 
single-distribution models would miss.

## Gaussian Mixture Models

A Gaussian Mixture Model (GMM) is a probabilistic model that assumes data are generated from a finite mixture of $k$ Gaussian components. Each component has its own mean $\mu_j$ and covariance $\Sigma_j$, and the overall distribution is a weighted combination of these individual Gaussians.

### Definition

Let $z$ denote the hidden (latent) variable indicating which component generated a data point. The prior probability of belonging to component $j$ is

$$
p(z = j) = \pi_j, \text{ with } \sum_{j = 1}^k \pi_j = 1.
$$

Given $z = j$, the observed variable $x$ follows a Gaussian distribution parameterized by that component:

$$
p(x \mid z = j) = \mathcal{N}(x \mid \mu_j, \Sigma_j).
$$

Because $z$ is unobserved, the marginal distribution of $x$ is obtained by summing over all possible components, yielding a weighted sum of Gaussians:

$$
p(x) = \sum_{j = 1}^k \pi_j \, \mathcal{N}(x \mid \mu_j, \Sigma_j).
$$

### Generative Process vs. Observation

We can view a GMM as defining a simple two-step generative process:

1. First, sample a component assignment $z \sim p(z)$.
2. Then, sample the observation conditioned on that assignment: $x \mid z \sim p(x \mid z)$.

In practice, however, we only observe the data $x$; the latent assignments $z$ are hidden from us. We need to infer these hidden states and estimate the model parameters from observed data.

## Maximum Likelihood Parameter Estimation

Suppose we have a dataset $X = \{x_1, \dots, x_n\}$ drawn from a mixture of $k$ Gaussian distributions. If we knew the cluster assignment $z_i \in \{1, \dots, k\}$ for each data point, we could treat the assignments as parameters and maximize the joint log-likelihood of the complete data $(x_i, z_i)$:

$$
\underset{\mu, \Sigma, z}{\mathrm{argmax}} \sum_{i = 1}^n \log p(x_i, z_i)
= \underset{\mu, \Sigma, z}{\mathrm{argmax}} \sum_{i = 1}^n \bigl(\log \pi_{z_i} + \log \mathcal{N}(x_i \mid \mu_{z_i}, \Sigma_{z_i})\bigr).
$$

To make the optimization more explicit, we can rewrite each $z_i$ using a one-hot encoding $z_{ij}$, where

$$
z_{ij} = \begin{cases}
1 & \text{if } z_i = j, \\
0 & \text{otherwise}.
\end{cases}
$$

Since every data point must belong to exactly one cluster, we have $\sum_{j=1}^k z_{ij} = 1$ for each $i$. Using this representation, the mixture probability and Gaussian density for a single point can be expressed as products over all components:

$$
\pi_{z_i} = \prod_{j = 1}^k \pi_j^{z_{ij}}, \qquad
\mathcal{N}(x_i \mid \mu_{z_i}, \Sigma_{z_i}) = \prod_{j = 1}^k \mathcal{N}(x_i \mid \mu_j, \Sigma_j)^{z_{ij}}.
$$

Substituting these into the objective gives a single double summation:

$$
\underset{\mu, \Sigma, z}{\mathrm{argmax}} \sum_{i = 1}^n \sum_{j = 1}^k z_{ij}\bigl(\log \pi_j + \log \mathcal{N}(x_i \mid \mu_j, \Sigma_j)\bigr).
$$

Unfortunately, we cannot maximize this directly because the latent assignments $z_{ij}$ are unknown. Optimizing over both the parameters and the discrete assignments simultaneously leads to a combinatorial problem with many local optima.

## Expectation-Maximization

The Expectation-Maximization (EM) algorithm is an iterative method for
finding maximum likelihood estimates (MLE) or maximum a posteriori (MAP)
estimates when data contain latent (hidden) variables or are
incomplete.

### Overview

Given observed data $X$, latent variables $Z$, and parameters $\theta$, we
want to maximize the marginal log-likelihood of the observed data:

$$
\log p(X \mid \theta) = \log \sum_{Z} p(X, Z \mid \theta).
$$

Because the summation sits inside the logarithm, the parameters are tightly
coupled and this objective is difficult to optimize directly. The key idea
behind EM is to construct a tractable lower bound and maximize the bound
instead.

To derive that bound, introduce an arbitrary distribution $q(Z)$ over the latent
variables. We can then rewrite:

$$
\log p(X \mid \theta) = \log \sum_{Z} q(Z) \frac{p(X, Z \mid \theta)}{q(Z)}
= \log \mathbb{E}_Z\!\left[\frac{p(X, Z \mid \theta)}{q(Z)}\right].
$$

According to Jensen's inequality, for a concave function $f$,

$$
f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)].
$$

![](jensens_inequality.png)

Since $\log$ is concave, we have:

$$
\log p(X \mid \theta) \ge \mathbb{E}_Z\!\left[\log \frac{p(X, Z \mid \theta)}{q(Z)}\right]
\triangleq \mathcal{L}(q, \theta).
$$

$\mathcal{L}(q, \theta)$ is called the **Evidence Lower Bound (ELBO)**.

Why is this a lower bound? The gap between the true log-likelihood and the ELBO turns out to be exactly the KL divergence between $q(Z)$ and the posterior $p(Z \mid X, \theta)$. To see why, take the difference:

$$
\log p(X \mid \theta) - \mathcal{L}(q, \theta)
= \mathbb{E}_Z\!\big[\log p(X \mid \theta)\big]
- \mathbb{E}_Z\!\left[\log \frac{p(X, Z \mid \theta)}{q(Z)}\right].
$$

The first expectation is just $\log p(X \mid \theta)$, because it does not depend on $Z$. Using the chain rule $p(X, Z \mid \theta) = p(Z \mid X, \theta)\,p(X \mid \theta)$, we have $\log p(X, Z \mid \theta) = \log p(Z \mid X, \theta) + \log p(X \mid \theta)$. Substituting this in and simplifying:

$$
\begin{aligned}
\log p(X \mid \theta) - \mathcal{L}(q, \theta)
&= \mathbb{E}_Z\!\left[
\log p(X \mid \theta)
- \log p(Z \mid X, \theta)
- \log p(X \mid \theta)
+ \log q(Z)
\right] \\
&= \mathbb{E}_Z\!\left[
\log q(Z) - \log p(Z \mid X, \theta)
\right] \\
&= \mathrm{KL}\big(q(Z) \,\|\, p(Z \mid X, \theta)\big).
\end{aligned}
$$

Rearranging gives the decomposition:

$$
\log p(X \mid \theta) = \mathcal{L}(q, \theta) + \mathrm{KL}\big(q(Z) \,\|\, p(Z \mid X, \theta)\big).
$$

The objective $\mathcal{L}(q, \theta)$ depends on two sets of variables: the model parameters $\theta$ and the distribution $q(Z)$. Optimizing them jointly is still difficult, but optimizing one while holding the other fixed is tractable. This suggests a coordinate-ascent strategy.

If we fix $\theta$, then $\log p(X \mid \theta)$ is constant. From the decomposition above, maximizing $\mathcal{L}(q, \theta)$ with respect to $q$ is equivalent to minimizing $\mathrm{KL}\big(q(Z) \,\|\, p(Z \mid X, \theta)\big)$. Because a KL divergence is always non-negative, the minimum is $0$, achieved when $q(Z) = p(Z \mid X, \theta)$. Setting $q$ to the posterior makes the bound tight: $\mathcal{L}(q, \theta) = \log p(X \mid \theta)$.

If we fix $q$, maximizing $\mathcal{L}(q, \theta)$ with respect to $\theta$ pushes the lower bound upward. Even though the bound may no longer be tight after $\theta$ changes, raising the bound is guaranteed to raise the true log-likelihood.

EM therefore alternates between two steps:

- **E-Step**: tightening the bound by optimizing $q$;
- **M-Step**: pushing the bound upward by optimizing $\theta$.

### $Q$-Function

First, let us expand the ELBO:

$$
\mathcal{L}(q, \theta)
= \mathbb{E}_Z\!\big[\log p(X, Z \mid \theta)\big] - \mathbb{E}_Z\!\big[\log q(Z)\big].
$$

The second term is the entropy of $q$, which does not depend on $\theta$ in the M-step because $q$ is fixed. Therefore, maximizing $\mathcal{L}$ with respect to $\theta$ is equivalent to maximizing only the first term.

Also note that the KL divergence vanishes in the E-step to make the bound tight:

$$
q(Z) = p(Z \mid X, \theta^{(t)}).
$$

Therefore, our goal becomes maximizing an expected complete-data log-likelihood, called **$Q$-function**:

$$
Q(\theta, \theta^{(t)}) = \mathbb{E}_{Z \mid X,\theta^{(t)}} \big[\log p(X, Z \mid \theta) \big].
$$

### E-Step

Fix $\theta = \theta^{(t)}$ and compute the $Q$-function.

How do we evaluate this expectation when $Z$ appears inside the logarithm? Because the expectation is over $Z$, the latent variable is averaged outside the log. 
For discrete $Z$, the expectation is a concrete weighted sum over all possible assignments:

$$
Q(\theta, \theta^{(t)}) = \sum_{Z} p(Z \mid X, \theta^{(t)}) \, \log p(X, Z \mid \theta).
$$

Each term evaluates $\log p(X, Z \mid \theta)$ at a particular assignment $Z$ and weights it by the posterior probability of that assignment. 
We never need to manipulate $q$ as an abstract distribution; we only need these posterior probabilities to form the weighted sum.

### M-Step

With $q(Z)$ fixed at the posterior from the E-step, maximize the $Q$-function with respect to $\theta$:

$$
\theta^{(t+1)} = \arg\max_{\theta} Q(\theta, \theta^{(t)}).
$$

This is usually a standard MLE problem, but the "data" now include soft assignments from the E-step.

### Monotonic Improvement

Each EM iteration guarantees a **non-decreasing observed log-likelihood**, illustrated below:

![](em_elbo.png)

1. **E-step**: Tightens the lower bound to touch $\log p(X \mid \theta^{(t)})$, and constructs a better ELBO.
2. **M-step**: Increases the lower bound to $\mathcal{L}(q, \theta^{(t+1)})$ (red arrows in the picture).

After an iteration:

$$
\mathcal{L}(q, \theta^{(t+1)}) \geq \mathcal{L}(q, \theta^{(t)}) = \log p(X \mid \theta^{(t)})
$$

The gap between the true likelihood and the ELBO is a KL divergence, which is always non-negative. Therefore:

$$
\log p(X \mid \theta^{(t+1)}) = \mathcal{L}(q, \theta^{(t+1)}) + \underbrace{\mathrm{KL}\big(q(Z) \,\|\, p(Z \mid X, \theta^{(t+1)})\big)}_{\geq 0} \geq \mathcal{L}(q, \theta^{(t+1)}).
$$

Chaining these together gives:

$$
\log p(X \mid \theta^{(t+1)}) \geq \log p(X \mid \theta^{(t)}).
$$

Thus, $\log p(X \mid \theta)$ climbs monotonically until convergence.

One caveat is that EM is only guaranteed to reach a *local* optimum, so initialization matters. A common heuristic is to seed the means with the centroids from a quick run of $K$-means. The algorithm can also converge arbitrarily slowly near saddle points or flat regions of the likelihood surface.

## Applying EM to GMM

We can now return to the mixture model from the beginning of this post. In the "Maximum Likelihood Parameter Estimation" section, we wrote the complete-data log-likelihood using hard one-hot assignments $z_{ij}$, but optimizing over both the parameters and these discrete indicators leads to a combinatorial problem with many local optima. EM solves exactly this difficulty.

For a Gaussian Mixture Model, the observed data are $X = \{x_1, \dots, x_N\}$, the latent assignments are $z_n \in \{1, \dots, K\}$, and the parameters are $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$. Using the same one-hot encoding $z_{nk} \in \{0, 1\}$ with $\sum_{k=1}^K z_{nk} = 1$ that we introduced earlier, the complete-data log-likelihood is:

$$
\log p(X, Z \mid \theta) = \sum_{n=1}^N \sum_{k=1}^K z_{nk} \bigl(\log \pi_k + \log \mathcal{N}(x_n \mid \mu_k, \Sigma_k)\bigr).
$$

Plugging this into the general $Q$-function gives the GMM-specific objective:

$$
\begin{aligned}
Q(\theta, \theta^{(t)}) &= \mathbb{E}_{Z \mid X,\theta^{(t)}} \big[\sum_{n=1}^N \sum_{k=1}^K z_{nk} \bigl(\log \pi_k + \log \mathcal{N}(x_n \mid \mu_k, \Sigma_k)\bigr)\big] \\
&= \sum_{n=1}^N \sum_{k=1}^K \mathbb{E}_{Z \mid X,\theta^{(t)}}[z_{nk}] \bigl(\log \pi_k + \log \mathcal{N}(x_n \mid \mu_k, \Sigma_k)\bigr).
\end{aligned}
$$

> Complete-data MLE pretends we know the latent assignments $z_{nk}$ and optimizes $\log p(X, Z \mid \theta)$ directly. Observed-data MLE, which is what EM targets, instead integrates over that uncertainty via marginalization: EM replaces each hard $z_{nk} \in \{0,1\}$ with its soft posterior expectation $\mathbb{E}[z_{nk}]$.

### E-Step: Compute Responsibilities

The E-step evaluates the expectations that appear in the $Q$-function above. Because $z_{nk}$ is an indicator, its conditional expectation is simply the posterior probability of belonging to component $k$:

$$
\mathbb{E}_{Z \mid X,\theta^{(t)}}[z_{nk}] = p(z_n = k \mid x_n, \theta^{(t)}) = \frac{\pi_k^{(t)} \, \mathcal{N}(x_n \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \, \mathcal{N}(x_n \mid \mu_j^{(t)}, \Sigma_j^{(t)})}.
$$

These posterior expectations are called responsibilities. They play the same role that the hard assignments $z_{nk}$ played in the complete-data MLE, but now as soft, fractional weights.

### M-Step: Weighted Maximum Likelihood

With the responsibilities fixed, maximizing $Q$ reduces to a weighted MLE problem for each Gaussian component:

$$
\begin{aligned}
N_k &= \sum_{n=1}^N \mathbb{E}[z_{nk}], \\[6pt]
\pi_k^{(t+1)} &= \frac{N_k}{N}, \\[6pt]
\mu_k^{(t+1)} &= \frac{1}{N_k} \sum_{n=1}^N \mathbb{E}[z_{nk}] \, x_n, \\[6pt]
\Sigma_k^{(t+1)} &= \frac{1}{N_k} \sum_{n=1}^N \mathbb{E}[z_{nk}] \, (x_n - \mu_k^{(t+1)})(x_n - \mu_k^{(t+1)})^\top.
\end{aligned}
$$

The algorithm alternates between these two steps until the log-likelihood or the parameters change by less than a small threshold $\epsilon$.

### Implementation in NumPy

Below is a compact, self-contained implementation using only NumPy.

```python
import numpy as np


class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.K = n_components      # number of mixture components
        self.max_iter = max_iter   # maximum EM iterations
        self.tol = tol             # convergence threshold for log-likelihood

    def _gaussian_pdf(self, X, mu, Sigma):
        """Evaluate the multivariate Gaussian N(mu, Sigma) for each row of X."""
        d = X.shape[1]
        det = np.linalg.det(Sigma)
        inv = np.linalg.inv(Sigma)
        diff = X - mu
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        return np.exp(exponent) / np.sqrt((2 * np.pi) ** d * det)

    def fit(self, X):
        N, D = X.shape

        # Initialize mixture weights uniformly and pick K random data points as means.
        # In practice, K-means centroids give a better starting point.
        self.pi = np.ones(self.K) / self.K
        self.mu = X[np.random.choice(N, self.K, replace=False)]
        self.Sigma = np.array([np.eye(D) for _ in range(self.K)])

        log_likelihoods = []

        for _ in range(self.max_iter):
            # --- E-step: compute responsibilities E[z_nk] ---
            # gamma[n, k] = pi_k * N(x_n | mu_k, Sigma_k) / sum_j(...)
            gamma = np.zeros((N, self.K))
            for k in range(self.K):
                gamma[:, k] = self.pi[k] * self._gaussian_pdf(
                    X, self.mu[k], self.Sigma[k]
                )
            gamma /= gamma.sum(axis=1, keepdims=True)

            # --- M-step: weighted maximum likelihood ---
            # N_k = sum_n E[z_nk]  (effective number of points in component k)
            Nk = gamma.sum(axis=0)

            # Update mixture weights: pi_k = N_k / N
            self.pi = Nk / N

            for k in range(self.K):
                # Weighted mean: mu_k = (1/N_k) sum_n E[z_nk] * x_n
                self.mu[k] = (gamma[:, k][:, None] * X).sum(axis=0) / Nk[k]

                diff = X - self.mu[k]
                # Weighted covariance: Sigma_k = (1/N_k) sum_n E[z_nk] * (x_n - mu_k)(x_n - mu_k)^T
                self.Sigma[k] = (gamma[:, k][:, None] * diff).T @ diff / Nk[k]

                # Small ridge to prevent singular covariance if a component collapses.
                self.Sigma[k] += 1e-6 * np.eye(D)

            # --- Compute observed-data log-likelihood for convergence check ---
            # log p(X | theta) = sum_n log sum_k pi_k * N(x_n | mu_k, Sigma_k)
            likelihood = np.zeros(N)
            for k in range(self.K):
                likelihood += self.pi[k] * self._gaussian_pdf(
                    X, self.mu[k], self.Sigma[k]
                )
            log_likelihoods.append(np.sum(np.log(likelihood)))

            if len(log_likelihoods) > 1:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                    break

        return self

    def predict(self, X):
        """Assign each point to the component with highest responsibility."""
        gamma = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * self._gaussian_pdf(
                X, self.mu[k], self.Sigma[k]
            )
        return gamma.argmax(axis=1)
```

