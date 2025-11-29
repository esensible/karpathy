# DAG Expression Evaluator

A cost-constrained, distribution-aware evaluation framework for expression trees in Go. This implementation uses the `expr-lang/expr` package for expression parsing and evaluation, combined with probabilistic reasoning and Value of Information (VoI) analysis for intelligent input selection.

## Overview

This framework treats expression evaluation as a sequential decision problem under uncertainty. Given:
- An expression DAG (Directed Acyclic Graph) with inputs, parameters, and expressions
- Costs associated with acquiring each input
- Prior probability distributions over unknown inputs

The evaluator decides which input to acquire next to minimize expected total cost while achieving a specified resolution objective.

## Features

- **Schema-driven DAG definition**: Define expression trees in JSON with inputs, parameters, enums, and expressions
- **Distribution types**: Uniform, Normal, Truncated Normal, Bernoulli, Categorical, and Empirical distributions
- **Monte Carlo propagation**: Forward propagation of uncertainty through the expression tree
- **Interval arithmetic**: Fast symbolic interval propagation for sensitivity analysis
- **Value of Information (VoI)**: Myopic VoI calculation for cost-aware input selection
- **Decision tables**: Support for FIRST and COLLECT hit policies
- **Conditional expressions**: Proper handling of ternary operators with branch pruning

## Installation

```bash
go get github.com/karpathy/dag-evaluator
```

## Usage

### Define a Schema

```json
{
  "schema_version": "1.2",
  "name": "SimpleComparison",
  "description": "Determine if A + B > 100",
  "root_node_id": "result",
  "enums": {},
  "parameters": {
    "threshold": {"description": "Threshold value", "value": 100}
  },
  "inputs": {
    "a": {
      "description": "First number",
      "question": "Enter A:",
      "label": "A",
      "output_type": "float",
      "acquisition_cost": 1.0,
      "prior": {"type": "uniform", "params": {"min": 0, "max": 100}}
    },
    "b": {
      "description": "Second number",
      "question": "Enter B:",
      "label": "B",
      "output_type": "float",
      "acquisition_cost": 2.0,
      "prior": {"type": "uniform", "params": {"min": 0, "max": 100}}
    }
  },
  "expressions": {
    "sum": {
      "description": "Sum of A and B",
      "output_type": "float",
      "expression": {"expression_type": "EXPR", "body": "a + b"}
    },
    "result": {
      "description": "Whether sum exceeds threshold",
      "output_type": "bool",
      "expression": {"expression_type": "EXPR", "body": "sum > threshold"}
    }
  }
}
```

### Programmatic Usage

```go
package main

import (
    "fmt"
    "math/rand"

    "github.com/karpathy/dag-evaluator/distribution"
    "github.com/karpathy/dag-evaluator/evaluator"
    "github.com/karpathy/dag-evaluator/schema"
    "github.com/karpathy/dag-evaluator/voi"
)

func main() {
    // Load schema
    s, err := schema.Load("schema.json")
    if err != nil {
        panic(err)
    }

    // Create evaluator
    cfg := evaluator.Config{
        NumSamples:    1000,
        NumVoISamples: 100,
        UtilityFunc:   voi.UtilityConfidence,
        CostWeight:    1.0,
        RNG:           rand.New(rand.NewSource(42)),
        Objective: voi.ResolutionObjective{
            Type:  voi.ResolveBoolean,
            Alpha: 0.95,
        },
    }

    eval, err := evaluator.New(s, cfg)
    if err != nil {
        panic(err)
    }

    // Initialize with priors
    eval.Initialize()

    // Get VoI scores for all inputs
    scores := eval.GetAllScores()
    for _, score := range scores {
        fmt.Printf("Input %s: VoI=%.4f, Cost=%.2f, Net=%.4f\n",
            score.InputID, score.VoI, score.Cost, score.NetScore)
    }

    // Acquire inputs
    eval.AcquireInput("a", distribution.NewFloatValue(60))
    eval.AcquireInput("b", distribution.NewFloatValue(50))

    // Get result
    result := eval.GetResult()
    fmt.Printf("Result: %s (cost: %.2f)\n", result.Value.AsString(), result.TotalCost)
}
```

### CLI Usage

```bash
# Build the CLI
go build -o dag-eval ./cmd/dag-eval

# Run interactively
./dag-eval -schema examples/loan_approval.json -interactive -verbose

# Run automatically with random inputs
./dag-eval -schema examples/simple_comparison.json -auto -verbose

# Show VoI scores
./dag-eval -schema examples/loan_approval.json -scores
```

## Architecture

```
dag-evaluator/
├── schema/          # Schema types and JSON loader
├── distribution/    # Probability distribution types
├── dag/             # DAG construction and node types
├── propagation/     # Distribution propagation (Monte Carlo + Interval)
├── voi/             # Value of Information calculator
├── evaluator/       # Main evaluation orchestration
├── cmd/dag-eval/    # CLI tool
└── examples/        # Example schemas
```

## Key Concepts

### Resolution Objectives

- `ResolveToValue`: Determine the exact value
- `ResolveToInterval`: Narrow to a specified interval width
- `ResolveGreaterThan`: P(Y > T) > alpha
- `ResolveLessThan`: P(Y < T) > alpha
- `ResolveBoolean`: Determine true/false with specified confidence

### Utility Functions

- `UtilityEntropy`: Minimize entropy (information-theoretic)
- `UtilityVariance`: Minimize variance (for numeric outputs)
- `UtilityConfidence`: Maximize classification confidence (for Boolean/categorical)
- `UtilityIntervalWidth`: Minimize interval width

### Prior Distributions

```json
{"type": "uniform", "params": {"min": 0, "max": 100}}
{"type": "normal", "params": {"mu": 50, "sigma": 10}}
{"type": "truncated_normal", "params": {"mu": 50, "sigma": 10, "lower": 0, "upper": 100}}
{"type": "bernoulli", "params": {"p": 0.5}}
{"type": "categorical", "params": {"values": ["A", "B", "C"], "probs": [0.5, 0.3, 0.2]}}
```

## Theoretical Foundation

This implementation is based on:

1. **Cost-Constrained, Distribution-Aware Evaluation Framework**: Integrates probabilistic priors, symbolic interval propagation, and cost-aware input selection.

2. **Myopic Value of Information Framework**: Uses greedy expected-future-cost minimization to select the next input to acquire.

Key principles:
- Evaluation (computation) is free; only input acquisition costs
- Uncertainty is modeled via probability distributions
- The goal is to resolve the root expression with minimum expected cost
- VoI guides input selection by comparing information gain against acquisition cost

## License

MIT
