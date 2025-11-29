// Package propagation provides distribution propagation through expression trees.
package propagation

import (
	"fmt"
	"math/rand"

	"github.com/expr-lang/expr"
	"github.com/karpathy/dag-evaluator/dag"
	"github.com/karpathy/dag-evaluator/distribution"
)

// Config holds configuration for the propagator.
type Config struct {
	// NumSamples is the number of Monte Carlo samples for distribution propagation.
	NumSamples int
	// RNG is the random number generator to use.
	RNG *rand.Rand
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		NumSamples: 1000,
		RNG:        rand.New(rand.NewSource(42)),
	}
}

// Propagator propagates distributions through a DAG.
type Propagator struct {
	config Config
	dag    *dag.DAG
	env    map[string]interface{}
}

// NewPropagator creates a new propagator.
func NewPropagator(d *dag.DAG, cfg Config) *Propagator {
	return &Propagator{
		config: cfg,
		dag:    d,
		env:    make(map[string]interface{}),
	}
}

// PropagateAll propagates distributions through all nodes in topological order.
func (p *Propagator) PropagateAll() error {
	// Build base environment with enums and parameters
	p.buildBaseEnvironment()

	// Process nodes in topological order
	for _, nodeID := range p.dag.TopoOrder {
		node, ok := p.dag.GetNode(nodeID)
		if !ok {
			continue
		}

		if err := p.propagateNode(node); err != nil {
			return fmt.Errorf("failed to propagate node '%s': %w", nodeID, err)
		}
	}

	return nil
}

// PropagateFrom propagates from a specific node (and all its dependencies).
func (p *Propagator) PropagateFrom(nodeID string) error {
	p.buildBaseEnvironment()

	// Find the node's position in topological order
	startIdx := -1
	for i, id := range p.dag.TopoOrder {
		if id == nodeID {
			startIdx = i
			break
		}
	}
	if startIdx < 0 {
		return fmt.Errorf("node '%s' not found in topological order", nodeID)
	}

	// Propagate from start to end
	for i := startIdx; i < len(p.dag.TopoOrder); i++ {
		node, ok := p.dag.GetNode(p.dag.TopoOrder[i])
		if !ok {
			continue
		}
		if err := p.propagateNode(node); err != nil {
			return fmt.Errorf("failed to propagate node '%s': %w", node.ID, err)
		}
	}

	return nil
}

func (p *Propagator) buildBaseEnvironment() {
	// Add enum types as maps
	for enumName, values := range p.dag.Enums {
		enumMap := make(map[string]string)
		for _, v := range values {
			enumMap[v] = v
		}
		p.env[enumName] = enumMap
	}

	// Add resolved parameters
	for id, node := range p.dag.Nodes {
		if node.IsParameter() && node.IsResolved {
			p.env[id] = node.Value.ToInterface()
		}
	}
}

func (p *Propagator) propagateNode(node *dag.Node) error {
	// Skip parameters (always resolved)
	if node.IsParameter() {
		return nil
	}

	// For input nodes, use their current distribution
	if node.IsInput() {
		if node.IsResolved {
			p.env[node.ID] = node.Value.ToInterface()
		}
		return nil
	}

	// For expression nodes, propagate through Monte Carlo
	if node.IsExpression() {
		return p.propagateExpression(node)
	}

	return nil
}

func (p *Propagator) propagateExpression(node *dag.Node) error {
	exprInfo := node.Expression
	if exprInfo == nil {
		return fmt.Errorf("expression node '%s' has no expression info", node.ID)
	}

	// Check if all dependencies are resolved (point distributions)
	allResolved := true
	for _, depID := range node.Dependencies {
		depNode, ok := p.dag.GetNode(depID)
		if !ok || !depNode.IsResolved {
			allResolved = false
			break
		}
	}

	if allResolved {
		// Evaluate directly
		return p.evaluateDeterministic(node)
	}

	// Use Monte Carlo propagation
	return p.propagateMonteCarlo(node)
}

func (p *Propagator) evaluateDeterministic(node *dag.Node) error {
	// Build environment with resolved values
	env := make(map[string]interface{})
	for k, v := range p.env {
		env[k] = v
	}
	for _, depID := range node.Dependencies {
		depNode, _ := p.dag.GetNode(depID)
		if depNode != nil && depNode.IsResolved {
			env[depID] = depNode.Value.ToInterface()
		}
	}

	// Evaluate the expression
	result, err := p.evaluateWithEnv(node, env)
	if err != nil {
		return err
	}

	// Convert result to Value and mark as resolved
	value := p.interfaceToValue(result)
	node.SetResolved(value)
	p.env[node.ID] = result

	return nil
}

func (p *Propagator) propagateMonteCarlo(node *dag.Node) error {
	samples := make([]distribution.Value, p.config.NumSamples)

	for i := 0; i < p.config.NumSamples; i++ {
		// Build environment with sampled values
		env := make(map[string]interface{})
		for k, v := range p.env {
			env[k] = v
		}

		// Sample from each dependency's distribution
		for _, depID := range node.Dependencies {
			depNode, ok := p.dag.GetNode(depID)
			if !ok {
				continue
			}

			if depNode.IsResolved {
				env[depID] = depNode.Value.ToInterface()
			} else if depNode.Distribution != nil {
				sample := depNode.Distribution.Sample(p.config.RNG)
				env[depID] = sample.ToInterface()
			}
		}

		// Evaluate with sampled environment
		result, err := p.evaluateWithEnv(node, env)
		if err != nil {
			// On error, use a default value
			samples[i] = distribution.Value{}
			continue
		}

		samples[i] = p.interfaceToValue(result)
	}

	// Create empirical distribution from samples
	empirical := distribution.NewEmpiricalDistribution(samples)
	node.SetDistribution(empirical)

	// If all samples are the same, mark as resolved
	if empirical.IsPoint() {
		if v, ok := empirical.PointValue(); ok {
			node.SetResolved(v)
			p.env[node.ID] = v.ToInterface()
		}
	}

	return nil
}

func (p *Propagator) evaluateWithEnv(node *dag.Node, env map[string]interface{}) (interface{}, error) {
	exprInfo := node.Expression

	if exprInfo.ExprType == "EXPR" {
		return p.evaluateExpr(exprInfo.Body, env)
	}

	if exprInfo.ExprType == "DECISION_TABLE" && exprInfo.Table != nil {
		return p.evaluateDecisionTable(exprInfo.Table, env)
	}

	return nil, fmt.Errorf("unknown expression type: %s", exprInfo.ExprType)
}

func (p *Propagator) evaluateExpr(body string, env map[string]interface{}) (interface{}, error) {
	program, err := expr.Compile(body, expr.Env(env), expr.AllowUndefinedVariables())
	if err != nil {
		return nil, fmt.Errorf("failed to compile expression: %w", err)
	}

	result, err := expr.Run(program, env)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate expression: %w", err)
	}

	return result, nil
}

func (p *Propagator) evaluateDecisionTable(table *dag.DecisionTableInfo, env map[string]interface{}) (interface{}, error) {
	switch table.HitPolicy {
	case "FIRST":
		for _, rule := range table.Rules {
			// Evaluate when condition
			matched, err := p.evaluateCondition(rule.WhenExpr, env)
			if err != nil {
				continue
			}
			if matched {
				// Evaluate and return then expression
				return p.evaluateExpr(rule.ThenExpr, env)
			}
		}
		return nil, fmt.Errorf("no rule matched in FIRST decision table")

	case "COLLECT":
		var results []interface{}
		for _, rule := range table.Rules {
			matched, err := p.evaluateCondition(rule.WhenExpr, env)
			if err != nil {
				continue
			}
			if matched {
				result, err := p.evaluateExpr(rule.ThenExpr, env)
				if err != nil {
					continue
				}
				results = append(results, result)
			}
		}
		return results, nil

	default:
		return nil, fmt.Errorf("unknown hit policy: %s", table.HitPolicy)
	}
}

func (p *Propagator) evaluateCondition(condition string, env map[string]interface{}) (bool, error) {
	result, err := p.evaluateExpr(condition, env)
	if err != nil {
		return false, err
	}

	switch v := result.(type) {
	case bool:
		return v, nil
	case int:
		return v != 0, nil
	case float64:
		return v != 0, nil
	case string:
		return v == "true", nil
	default:
		return false, fmt.Errorf("condition did not evaluate to boolean: %T", result)
	}
}

func (p *Propagator) interfaceToValue(v interface{}) distribution.Value {
	switch val := v.(type) {
	case float64:
		return distribution.NewFloatValue(val)
	case float32:
		return distribution.NewFloatValue(float64(val))
	case int:
		return distribution.NewIntValue(int64(val))
	case int64:
		return distribution.NewIntValue(val)
	case int32:
		return distribution.NewIntValue(int64(val))
	case bool:
		return distribution.NewBoolValue(val)
	case string:
		return distribution.NewStringValue(val)
	case map[string]interface{}:
		obj := make(map[string]distribution.Value)
		for k, v := range val {
			obj[k] = p.interfaceToValue(v)
		}
		return distribution.NewObjectValue(obj)
	default:
		// Try to handle as float
		if f, ok := toFloat64(v); ok {
			return distribution.NewFloatValue(f)
		}
		return distribution.NewStringValue(fmt.Sprintf("%v", v))
	}
}

func toFloat64(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case int32:
		return float64(val), true
	case uint:
		return float64(val), true
	case uint64:
		return float64(val), true
	case uint32:
		return float64(val), true
	default:
		return 0, false
	}
}

// GetRootDistribution returns the current distribution at the root node.
func (p *Propagator) GetRootDistribution() distribution.Distribution {
	root := p.dag.GetRoot()
	if root == nil {
		return nil
	}
	return root.Distribution
}

// SampleWithObservation samples the DAG with a hypothetical observation.
// Returns the resulting root distribution without modifying the DAG.
func (p *Propagator) SampleWithObservation(inputID string, value distribution.Value, numSamples int) (distribution.Distribution, error) {
	// Clone the DAG
	clonedDAG := p.dag.Clone()

	// Set the observation
	node, ok := clonedDAG.GetNode(inputID)
	if !ok {
		return nil, fmt.Errorf("input '%s' not found", inputID)
	}
	node.SetResolved(value)

	// Create new propagator for cloned DAG
	cfg := p.config
	cfg.NumSamples = numSamples
	clonedProp := NewPropagator(clonedDAG, cfg)

	// Propagate
	if err := clonedProp.PropagateAll(); err != nil {
		return nil, err
	}

	return clonedProp.GetRootDistribution(), nil
}
