# STaR: Self-taught Reasoner

⚠️ repo is lwk under construction rn so it's not really runnable sorry :(

paper replication attempt of "STaR: Bootstrapping Reasoning With Reasoning" (https://arxiv.org/abs/2203.14465). educational purposes only :p 

## Installation

```bash
pip install -r requirements.txt
```

**tl;dr**  
STaR is a way to improve base language models correctness against reasoning questions by iteratively bootstrapping model-generated rationales that lead to correct answers (including rationales conditioned on the correct answer when the model initially fails), and using these rationales as training data



## how it works


Procure a base dataset containing Question - Answer pair

$$D = \{(x_i, y_i)\}_{i=1}^{D}$$


Where:
- $x_i$ is a question (or problem),
- $y_i$ is the correct answer.

in practice, the dataset is split into:
- $D_{train}$: used for STaR training,
- $D_{dev}$: used for validation and early stopping,
- $D_{test}$: used only for final evaluation.

Only $D_{train}$ participates in the STaR bootstrapping loop.

Starting w/ a pretrained model $M_0$ we run multiple rounds of SFT w/ bootstrapped rationales where for a single loop, we:

1. Run inference against question $x_i$ from $D_{train}$ to produce rationale $\hat{r}$ leading to answer $\hat{y}$ (for $\forall x_i \in D_{train}$), using model $M_{i-1}$ .

2.  
   i. For, $(x_i, \hat{r}, \hat{y})$ where $\hat{y} = y$
   we store all valid question–rationale–answer triplets outputted by the model in dataset $O$.  

   ii. For $(x_i, \hat{r}, \hat{y})$ where $\hat{y} \neq y$, we give the model $(x, y)$ with $y$ as a hint for the model to generate a rationalization. If the model correctly generates a rationale for the correct answer, we save this as $(x, \hat{r}, y)$ into $O$.

3. Fine-tune model $M_i$ via SFT on $O$.

4. repeatedly add to and train model with $O$ until performance plateaus

Let $M_i$ denote a new model instance on the $i$-th STaR iteration.

$M_{i-1}$ depicts the state of the model AFTER fine-tuning occurs. 





Before starting any training loop, we build $P$, a subset of $D$ (where $P \ll D$) used as working examples for prompting:

$$P = \{(x_i^p, r_i^p, y_i^p)\}_{i=1}^{P}$$ 

where
- $x_i^p$ is a single example's input (question/problem)
- $r_i^p$ is its intermediate rationale (the step-by-step reasoning / explanation)  
- $y_i^p$ is its final answer/label

Then for each $x_i$ in $D_{train}$ which we run inference on, we build a single prompt by concatenating:

$$(x_1^p, r_1^p, y_1^p, \ldots, x_P^p, r_P^p, y_P^p, x_i)$$

<br><br><br><br><br>



I use wandb for experiment tracking: initializing runs, logging training and validation metrics (losses, performance numbers), logging technical stats (grad-noise, step/tokens/sec, grad norms), and resuming runs during inference/eval.







Special thanks to:


[![YouTube Video](https://img.youtube.com/vi/rJkTsNrnu8g/0.jpg)](https://www.youtube.com/watch?v=rJkTsNrnu8g)


