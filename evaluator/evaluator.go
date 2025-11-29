// Package evaluator provides the main orchestration for DAG evaluation.
package evaluator

import (
	"fmt"
	"math/rand"

	"github.com/karpathy/dag-evaluator/dag"
	"github.com/karpathy/dag-evaluator/distribution"
	"github.com/karpathy/dag-evaluator/propagation"
	"github.com/karpathy/dag-evaluator/schema"
	"github.com/karpathy/dag-evaluator/voi"
)

// Config holds configuration for the evaluator.
type Config struct {
	// NumSamples for Monte Carlo propagation.
	NumSamples int
	// NumVoISamples for VoI calculation.
	NumVoISamples int
	// UtilityFunc for scoring inputs.
	UtilityFunc voi.UtilityFunction
	// CostWeight (lambda) for cost-benefit tradeoff.
	CostWeight float64
	// MaxIterations limits the evaluation loop.
	MaxIterations int
	// RNG is the random number generator.
	RNG *rand.Rand
	// Resolution objective
	Objective voi.ResolutionObjective
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		NumSamples:    1000,
		NumVoISamples: 100,
		UtilityFunc:   voi.UtilityConfidence,
		CostWeight:    1.0,
		MaxIterations: 100,
		RNG:           rand.New(rand.NewSource(42)),
		Objective: voi.ResolutionObjective{
			Type:  voi.ResolveBoolean,
			Alpha: 0.95,
		},
	}
}

// Evaluator orchestrates the cost-aware evaluation of expression DAGs.
type Evaluator struct {
	config     Config
	schema     *schema.Schema
	dag        *dag.DAG
	propagator *propagation.Propagator
	voiCalc    *voi.Calculator
	history    []EvaluationStep
	totalCost  float64
}

// EvaluationStep records a single step in the evaluation process.
type EvaluationStep struct {
	StepNumber    int
	InputAcquired string
	InputValue    distribution.Value
	Cost          float64
	TotalCost     float64
	VoIScore      float64
	Confidence    float64
	IsResolved    bool
}

// Result holds the final evaluation result.
type Result struct {
	// Value is the resolved root value (if resolved).
	Value distribution.Value
	// Distribution is the final distribution at the root.
	Distribution distribution.Distribution
	// IsResolved indicates if the resolution objective was met.
	IsResolved bool
	// Confidence is the final confidence level.
	Confidence float64
	// TotalCost is the total acquisition cost incurred.
	TotalCost float64
	// Steps is the number of inputs acquired.
	Steps int
	// History contains all evaluation steps.
	History []EvaluationStep
}

// New creates a new Evaluator from a schema.
func New(s *schema.Schema, cfg Config) (*Evaluator, error) {
	// Build DAG from schema
	d, err := dag.BuildFromSchema(s)
	if err != nil {
		return nil, fmt.Errorf("failed to build DAG: %w", err)
	}

	// Create propagator
	propConfig := propagation.Config{
		NumSamples: cfg.NumSamples,
		RNG:        cfg.RNG,
	}
	prop := propagation.NewPropagator(d, propConfig)

	// Create VoI calculator
	voiConfig := voi.Config{
		NumVoISamples:  cfg.NumVoISamples,
		NumPropSamples: cfg.NumSamples / 2,
		UtilityFunc:    cfg.UtilityFunc,
		CostWeight:     cfg.CostWeight,
		RNG:            cfg.RNG,
	}
	calc := voi.NewCalculator(d, voiConfig)

	return &Evaluator{
		config:     cfg,
		schema:     s,
		dag:        d,
		propagator: prop,
		voiCalc:    calc,
		history:    make([]EvaluationStep, 0),
	}, nil
}

// NewFromDAG creates an Evaluator from an existing DAG.
func NewFromDAG(d *dag.DAG, cfg Config) *Evaluator {
	propConfig := propagation.Config{
		NumSamples: cfg.NumSamples,
		RNG:        cfg.RNG,
	}
	prop := propagation.NewPropagator(d, propConfig)

	voiConfig := voi.Config{
		NumVoISamples:  cfg.NumVoISamples,
		NumPropSamples: cfg.NumSamples / 2,
		UtilityFunc:    cfg.UtilityFunc,
		CostWeight:     cfg.CostWeight,
		RNG:            cfg.RNG,
	}
	calc := voi.NewCalculator(d, voiConfig)

	return &Evaluator{
		config:     cfg,
		dag:        d,
		propagator: prop,
		voiCalc:    calc,
		history:    make([]EvaluationStep, 0),
	}
}

// Initialize performs initial propagation with priors.
func (e *Evaluator) Initialize() error {
	return e.propagator.PropagateAll()
}

// GetNextInput returns the next input to acquire based on VoI analysis.
func (e *Evaluator) GetNextInput() (*voi.InputScore, bool) {
	return e.voiCalc.SelectNextInput()
}

// GetAllScores returns VoI scores for all unresolved inputs.
func (e *Evaluator) GetAllScores() []voi.InputScore {
	return e.voiCalc.ComputeAllScores()
}

// AcquireInput records an acquired input value and re-propagates.
func (e *Evaluator) AcquireInput(inputID string, value distribution.Value) error {
	node, ok := e.dag.GetNode(inputID)
	if !ok {
		return fmt.Errorf("input '%s' not found", inputID)
	}
	if !node.IsInput() {
		return fmt.Errorf("node '%s' is not an input", inputID)
	}

	// Record the acquisition
	cost := node.GetCost()
	e.totalCost += cost

	// Set the value
	node.SetResolved(value)

	// Re-propagate
	if err := e.propagator.PropagateAll(); err != nil {
		return fmt.Errorf("failed to propagate after acquisition: %w", err)
	}

	// Record step
	step := EvaluationStep{
		StepNumber:    len(e.history) + 1,
		InputAcquired: inputID,
		InputValue:    value,
		Cost:          cost,
		TotalCost:     e.totalCost,
		VoIScore:      0, // Could capture the pre-acquisition score
		Confidence:    e.voiCalc.GetCurrentConfidence(),
		IsResolved:    e.voiCalc.IsResolved(e.config.Objective),
	}
	e.history = append(e.history, step)

	return nil
}

// IsResolved checks if the resolution objective is met.
func (e *Evaluator) IsResolved() bool {
	return e.voiCalc.IsResolved(e.config.Objective)
}

// GetCurrentConfidence returns the current confidence level.
func (e *Evaluator) GetCurrentConfidence() float64 {
	return e.voiCalc.GetCurrentConfidence()
}

// GetRootDistribution returns the current distribution at the root.
func (e *Evaluator) GetRootDistribution() distribution.Distribution {
	root := e.dag.GetRoot()
	if root == nil {
		return nil
	}
	return root.Distribution
}

// GetRootValue returns the resolved root value if available.
func (e *Evaluator) GetRootValue() (distribution.Value, bool) {
	root := e.dag.GetRoot()
	if root == nil {
		return distribution.Value{}, false
	}
	if root.IsResolved {
		return root.Value, true
	}
	if root.Distribution != nil && root.Distribution.IsPoint() {
		return root.Distribution.PointValue()
	}
	return distribution.Value{}, false
}

// GetTotalCost returns the total acquisition cost so far.
func (e *Evaluator) GetTotalCost() float64 {
	return e.totalCost
}

// GetHistory returns the evaluation history.
func (e *Evaluator) GetHistory() []EvaluationStep {
	return e.history
}

// GetDAG returns the underlying DAG.
func (e *Evaluator) GetDAG() *dag.DAG {
	return e.dag
}

// GetInput returns information about an input node.
func (e *Evaluator) GetInput(inputID string) (*dag.Node, bool) {
	node, ok := e.dag.GetNode(inputID)
	if !ok || !node.IsInput() {
		return nil, false
	}
	return node, true
}

// GetUnresolvedInputs returns all unresolved input nodes.
func (e *Evaluator) GetUnresolvedInputs() []*dag.Node {
	return e.dag.GetUnresolvedInputNodes()
}

// RunAutomatic runs the evaluation automatically until resolved or budget exhausted.
// The inputProvider function is called to get values for inputs.
func (e *Evaluator) RunAutomatic(inputProvider func(inputID string, node *dag.Node) (distribution.Value, error), budget float64) (*Result, error) {
	// Initialize
	if err := e.Initialize(); err != nil {
		return nil, err
	}

	for i := 0; i < e.config.MaxIterations; i++ {
		// Check if resolved
		if e.IsResolved() {
			break
		}

		// Check budget
		if budget > 0 && e.totalCost >= budget {
			break
		}

		// Get next input
		score, found := e.GetNextInput()
		if !found {
			break
		}

		// Check if acquisition is worth it
		// Only stop if net score is very negative AND we already have decent confidence
		if score.NetScore < -score.Cost && e.GetCurrentConfidence() > 0.8 {
			// Not worth acquiring more inputs - cost outweighs any possible benefit
			break
		}

		// Check budget constraint
		if budget > 0 && e.totalCost+score.Cost > budget {
			break
		}

		// Get the input value
		node, _ := e.dag.GetNode(score.InputID)
		value, err := inputProvider(score.InputID, node)
		if err != nil {
			// Skip this input
			continue
		}

		// Acquire the input
		if err := e.AcquireInput(score.InputID, value); err != nil {
			return nil, err
		}
	}

	return e.GetResult(), nil
}

// GetResult returns the current evaluation result.
func (e *Evaluator) GetResult() *Result {
	result := &Result{
		Distribution: e.GetRootDistribution(),
		IsResolved:   e.IsResolved(),
		Confidence:   e.GetCurrentConfidence(),
		TotalCost:    e.totalCost,
		Steps:        len(e.history),
		History:      e.history,
	}

	if value, ok := e.GetRootValue(); ok {
		result.Value = value
	}

	return result
}

// SimulateAcquisition simulates acquiring an input without modifying the evaluator.
// Returns the expected result after acquisition.
func (e *Evaluator) SimulateAcquisition(inputID string, value distribution.Value) (*Result, error) {
	// Clone the DAG
	clonedDAG := e.dag.Clone()

	// Create new evaluator on cloned DAG
	clonedEval := NewFromDAG(clonedDAG, e.config)
	clonedEval.totalCost = e.totalCost

	// Acquire the input
	if err := clonedEval.AcquireInput(inputID, value); err != nil {
		return nil, err
	}

	return clonedEval.GetResult(), nil
}

// EstimateRemainingCost estimates the cost to complete resolution.
func (e *Evaluator) EstimateRemainingCost() float64 {
	return e.voiCalc.GetExpectedTotalCost(e.config.Objective)
}

// InputProvider is a function type for providing input values.
type InputProvider func(inputID string, node *dag.Node) (distribution.Value, error)

// InteractiveProvider creates an input provider that uses a callback for each input.
func InteractiveProvider(askFunc func(inputID string, question string, label string) (interface{}, error)) InputProvider {
	return func(inputID string, node *dag.Node) (distribution.Value, error) {
		question := ""
		label := inputID
		if node.Input != nil {
			question = node.Input.Question
			if node.Input.Label != "" {
				label = node.Input.Label
			}
		}

		rawValue, err := askFunc(inputID, question, label)
		if err != nil {
			return distribution.Value{}, err
		}

		// Convert raw value to distribution.Value
		switch v := rawValue.(type) {
		case float64:
			return distribution.NewFloatValue(v), nil
		case float32:
			return distribution.NewFloatValue(float64(v)), nil
		case int:
			return distribution.NewIntValue(int64(v)), nil
		case int64:
			return distribution.NewIntValue(v), nil
		case bool:
			return distribution.NewBoolValue(v), nil
		case string:
			if node.Input != nil && node.Input.EnumType != "" {
				return distribution.NewEnumValue(node.Input.EnumType, v), nil
			}
			return distribution.NewStringValue(v), nil
		default:
			return distribution.NewStringValue(fmt.Sprintf("%v", v)), nil
		}
	}
}

// RandomProvider creates an input provider that samples from the prior.
func RandomProvider(rng *rand.Rand) InputProvider {
	return func(inputID string, node *dag.Node) (distribution.Value, error) {
		if node.Distribution != nil {
			return node.Distribution.Sample(rng), nil
		}
		if node.Input != nil && node.Input.Prior != nil {
			return node.Input.Prior.Sample(rng), nil
		}
		return distribution.NewFloatValue(rng.Float64()), nil
	}
}

// FixedProvider creates an input provider from a map of fixed values.
func FixedProvider(values map[string]interface{}) InputProvider {
	return func(inputID string, node *dag.Node) (distribution.Value, error) {
		v, ok := values[inputID]
		if !ok {
			return distribution.Value{}, fmt.Errorf("no value provided for input '%s'", inputID)
		}

		switch val := v.(type) {
		case float64:
			return distribution.NewFloatValue(val), nil
		case float32:
			return distribution.NewFloatValue(float64(val)), nil
		case int:
			return distribution.NewIntValue(int64(val)), nil
		case int64:
			return distribution.NewIntValue(val), nil
		case bool:
			return distribution.NewBoolValue(val), nil
		case string:
			if node.Input != nil && node.Input.EnumType != "" {
				return distribution.NewEnumValue(node.Input.EnumType, val), nil
			}
			return distribution.NewStringValue(val), nil
		case distribution.Value:
			return val, nil
		default:
			return distribution.NewStringValue(fmt.Sprintf("%v", v)), nil
		}
	}
}
