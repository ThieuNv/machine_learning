Cach chon Activation: https://stackoverflow.com/questions/34229140/choosing-from-different-cost-function-and-activation-function-of-a-neural-networ


What to use. Now to the last question, how does one choose which activation and cost functions to use. These advices will work for majority of cases:

If you do classification, use softmax for the last layer's nonlinearity and cross entropy as a cost function.
If you do regression, use sigmoid or tanh for the last layer's nonlinearity and squared error as a cost function.
Use ReLU as a nonlienearity between layers.
Use better optimizers (AdamOptimizer, AdagradOptimizer) instead of GradientDescentOptimizer, or use momentum for faster convergence,


Link youtube: https://www.youtube.com/watch?v=-7scQpJT7uo

From experience I'd recommend in order, ELU (exponential linear units) >> leaky ReLU > ReLU > tanh, sigmoid.
I agree that you basically never have an excuse to use tanh or sigmoid.﻿



