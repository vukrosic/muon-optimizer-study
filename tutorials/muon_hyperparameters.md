# Muon Optimizer Hyperparameters

Understanding the hyperparameters of the Muon optimizer is crucial for effective tuning and achieving optimal performance. This section details the key parameters explored in the ablation study.

## 1. Learning Rate (LR)

The learning rate is one of the most critical hyperparameters. It determines the step size at each iteration while moving towards a minimum of the loss function.
*   **High LR:** Can lead to faster convergence but might overshoot the minimum, causing oscillations or divergence.
*   **Low LR:** Ensures more stable convergence but can be very slow, potentially getting stuck in local minima.

In this study, we observed the impact of varying the learning rate (e.g., 0.005, 0.015, 0.02) on the Muon optimizer's performance.

## 2. Momentum

Momentum helps accelerate SGD in the relevant direction and dampens oscillations. It adds a fraction of the update vector of the past time step to the current update vector.
*   **High Momentum:** Can help overcome local minima and speed up convergence in consistent directions.
*   **Low Momentum:** Reduces the "smoothing" effect of past updates.

The study investigated different momentum values (e.g., 0.88, 0.90, 0.92, 0.93, 0.95, 0.96) to find the optimal balance.

## 3. Newton-Schulz (NS) Steps

The Newton-Schulz iteration is a method used to approximate the inverse of a matrix. In the context of optimizers like Muon, it might be used to approximate second-order information (like the inverse Hessian) to guide the optimization process.
*   **More NS Steps:** Can lead to a more accurate approximation of the inverse, potentially resulting in more precise and efficient updates. However, it also increases computational cost per step.
*   **Fewer NS Steps:** Reduces computational overhead but might lead to less accurate updates.

This ablation study varied the number of Newton-Schulz steps (e.g., 6, 8, 9, 10, 12) to understand its effect on convergence and performance. Different "aggressiveness" levels (Conservative, Mild Aggressive, Ultra Stable) likely refer to how these NS steps are applied or combined with other techniques.

## 4. Nesterov Momentum

Nesterov Accelerated Gradient (NAG) is a variant of momentum that looks ahead. Instead of calculating the gradient at the current position, it calculates it at an approximate future position. This often leads to faster convergence and better stability.
*   **With Nesterov:** Generally provides a stronger "look-ahead" capability, often leading to better performance.
*   **Without Nesterov:** Standard momentum behavior.

The study included a comparison of Muon with and without Nesterov momentum to assess its contribution.

By systematically varying these hyperparameters, the ablation study provides valuable insights into the Muon optimizer's sensitivity and optimal configurations for the given task.
