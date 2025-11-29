package propagation

import (
	"fmt"
	"math"

	"github.com/expr-lang/expr"
	"github.com/expr-lang/expr/ast"
	"github.com/karpathy/dag-evaluator/dag"
	"github.com/karpathy/dag-evaluator/distribution"
)

// IntervalPropagator propagates intervals through the DAG using interval arithmetic.
// This is faster than Monte Carlo but less precise for non-monotonic operations.
type IntervalPropagator struct {
	dag *dag.DAG
}

// NewIntervalPropagator creates a new interval propagator.
func NewIntervalPropagator(d *dag.DAG) *IntervalPropagator {
	return &IntervalPropagator{dag: d}
}

// PropagateIntervals propagates intervals through all nodes.
func (p *IntervalPropagator) PropagateIntervals() error {
	for _, nodeID := range p.dag.TopoOrder {
		node, ok := p.dag.GetNode(nodeID)
		if !ok {
			continue
		}

		if err := p.propagateNodeInterval(node); err != nil {
			return fmt.Errorf("failed to propagate interval for '%s': %w", nodeID, err)
		}
	}
	return nil
}

func (p *IntervalPropagator) propagateNodeInterval(node *dag.Node) error {
	if node.IsParameter() {
		// Parameters have point intervals
		if f, ok := node.Value.AsFloat(); ok {
			node.Interval = distribution.Point(f)
		}
		return nil
	}

	if node.IsInput() {
		// Use distribution support as interval
		node.Interval = distribution.IntervalFromDistribution(node.Distribution)
		return nil
	}

	if node.IsExpression() {
		return p.propagateExpressionInterval(node)
	}

	return nil
}

func (p *IntervalPropagator) propagateExpressionInterval(node *dag.Node) error {
	exprInfo := node.Expression
	if exprInfo == nil {
		return nil
	}

	// Build interval environment from all nodes (including parameters)
	intervals := make(map[string]distribution.Interval)
	for id, n := range p.dag.Nodes {
		if !n.Interval.IsEmpty() {
			intervals[id] = n.Interval
		} else if n.IsParameter() && n.IsResolved {
			// Parameters should have point intervals from their value
			if f, ok := n.Value.AsFloat(); ok {
				intervals[id] = distribution.Point(f)
			}
		}
	}

	if exprInfo.ExprType == "EXPR" {
		interval, err := p.evaluateIntervalExpr(exprInfo.Body, intervals)
		if err != nil {
			// Fall back to entire interval on error
			node.Interval = distribution.Entire()
			return nil
		}
		node.Interval = interval
		return nil
	}

	if exprInfo.ExprType == "DECISION_TABLE" && exprInfo.Table != nil {
		interval, err := p.evaluateDecisionTableInterval(exprInfo.Table, intervals)
		if err != nil {
			node.Interval = distribution.Entire()
			return nil
		}
		node.Interval = interval
		return nil
	}

	return nil
}

func (p *IntervalPropagator) evaluateIntervalExpr(body string, intervals map[string]distribution.Interval) (distribution.Interval, error) {
	program, err := expr.Compile(body, expr.AllowUndefinedVariables())
	if err != nil {
		return distribution.Entire(), err
	}

	return p.evaluateASTInterval(program.Node(), intervals)
}

func (p *IntervalPropagator) evaluateASTInterval(node ast.Node, intervals map[string]distribution.Interval) (distribution.Interval, error) {
	switch n := node.(type) {
	case *ast.IntegerNode:
		return distribution.Point(float64(n.Value)), nil

	case *ast.FloatNode:
		return distribution.Point(n.Value), nil

	case *ast.IdentifierNode:
		if interval, ok := intervals[n.Value]; ok {
			return interval, nil
		}
		// Unknown variable - return entire interval
		return distribution.Entire(), nil

	case *ast.BinaryNode:
		left, err := p.evaluateASTInterval(n.Left, intervals)
		if err != nil {
			return distribution.Entire(), err
		}
		right, err := p.evaluateASTInterval(n.Right, intervals)
		if err != nil {
			return distribution.Entire(), err
		}

		switch n.Operator {
		case "+":
			return left.Add(right), nil
		case "-":
			return left.Sub(right), nil
		case "*":
			return left.Mul(right), nil
		case "/":
			return left.Div(right), nil
		case "%":
			return left.Mod(right), nil
		case "<", ">", "<=", ">=", "==", "!=", "&&", "||":
			// Boolean operations return [0, 1]
			return distribution.NewInterval(0, 1), nil
		default:
			return distribution.Entire(), nil
		}

	case *ast.UnaryNode:
		operand, err := p.evaluateASTInterval(n.Node, intervals)
		if err != nil {
			return distribution.Entire(), err
		}

		switch n.Operator {
		case "-":
			return operand.Neg(), nil
		case "!":
			return distribution.NewInterval(0, 1), nil
		default:
			return operand, nil
		}

	case *ast.ConditionalNode:
		// For ternary, union of both branches
		exp1, err := p.evaluateASTInterval(n.Exp1, intervals)
		if err != nil {
			exp1 = distribution.Entire()
		}
		exp2, err := p.evaluateASTInterval(n.Exp2, intervals)
		if err != nil {
			exp2 = distribution.Entire()
		}
		return exp1.Union(exp2), nil

	case *ast.MemberNode:
		// Member access - need to handle specially
		return distribution.Entire(), nil

	case *ast.CallNode:
		// Function calls - handle common math functions
		return p.evaluateCallInterval(n, intervals)

	default:
		return distribution.Entire(), nil
	}
}

func (p *IntervalPropagator) evaluateCallInterval(node *ast.CallNode, intervals map[string]distribution.Interval) (distribution.Interval, error) {
	// Get function name
	var funcName string
	if ident, ok := node.Callee.(*ast.IdentifierNode); ok {
		funcName = ident.Value
	} else {
		return distribution.Entire(), nil
	}

	if len(node.Arguments) == 0 {
		return distribution.Entire(), nil
	}

	arg, err := p.evaluateASTInterval(node.Arguments[0], intervals)
	if err != nil {
		return distribution.Entire(), err
	}

	switch funcName {
	case "abs":
		return arg.Abs(), nil
	case "sqrt":
		return arg.Sqrt(), nil
	case "exp":
		return arg.Exp(), nil
	case "log", "ln":
		return arg.Log(), nil
	case "min":
		if len(node.Arguments) >= 2 {
			arg2, _ := p.evaluateASTInterval(node.Arguments[1], intervals)
			return distribution.NewInterval(
				math.Min(arg.Lower, arg2.Lower),
				math.Min(arg.Upper, arg2.Upper),
			), nil
		}
	case "max":
		if len(node.Arguments) >= 2 {
			arg2, _ := p.evaluateASTInterval(node.Arguments[1], intervals)
			return distribution.NewInterval(
				math.Max(arg.Lower, arg2.Lower),
				math.Max(arg.Upper, arg2.Upper),
			), nil
		}
	}

	return distribution.Entire(), nil
}

func (p *IntervalPropagator) evaluateDecisionTableInterval(table *dag.DecisionTableInfo, intervals map[string]distribution.Interval) (distribution.Interval, error) {
	// For FIRST policy, take union of all possible then values
	// For COLLECT, same approach
	result := distribution.Empty()

	for _, rule := range table.Rules {
		thenInterval, err := p.evaluateIntervalExpr(rule.ThenExpr, intervals)
		if err != nil {
			continue
		}
		result = result.Union(thenInterval)
	}

	if result.IsEmpty() {
		return distribution.Entire(), nil
	}
	return result, nil
}

// CanAffectRoot determines if changing an input can possibly affect the root value.
func (p *IntervalPropagator) CanAffectRoot(inputID string) bool {
	// Check if the input is in the dependency chain of the root
	root := p.dag.GetRoot()
	if root == nil {
		return false
	}

	visited := make(map[string]bool)
	return p.isInDependencyChain(root.ID, inputID, visited)
}

func (p *IntervalPropagator) isInDependencyChain(nodeID, targetID string, visited map[string]bool) bool {
	if nodeID == targetID {
		return true
	}
	if visited[nodeID] {
		return false
	}
	visited[nodeID] = true

	node, ok := p.dag.GetNode(nodeID)
	if !ok {
		return false
	}

	for _, depID := range node.Dependencies {
		if p.isInDependencyChain(depID, targetID, visited) {
			return true
		}
	}
	return false
}

// ComputeSensitivity computes how sensitive the root is to each input.
// For numeric roots, this is interval width reduction.
// For boolean roots, this is based on dependency chain membership.
func (p *IntervalPropagator) ComputeSensitivity() map[string]float64 {
	sensitivity := make(map[string]float64)

	root := p.dag.GetRoot()
	if root == nil {
		return sensitivity
	}

	// Check if root is Boolean (interval is [0, 1])
	isBooleanRoot := root.Interval.Lower == 0 && root.Interval.Upper == 1 &&
		root.Interval.Width() == 1

	baseWidth := root.Interval.Width()
	if math.IsInf(baseWidth, 0) || math.IsNaN(baseWidth) {
		baseWidth = 1e12 // Large but finite
	}

	for _, node := range p.dag.GetAllInputNodes() {
		if node.IsResolved {
			sensitivity[node.ID] = 0
			continue
		}

		// For Boolean roots, use dependency-based sensitivity
		if isBooleanRoot {
			if p.CanAffectRoot(node.ID) {
				// Assign sensitivity based on relative interval width of the input
				inputWidth := node.Interval.Width()
				if math.IsInf(inputWidth, 0) || math.IsNaN(inputWidth) {
					inputWidth = 1e6
				}
				// Normalize to [0, 1] - wider inputs have more potential impact
				sensitivity[node.ID] = math.Min(1.0, inputWidth/1e6)
				if sensitivity[node.ID] < 0.1 {
					sensitivity[node.ID] = 0.1 // Minimum sensitivity if in dependency chain
				}
			} else {
				sensitivity[node.ID] = 0
			}
			continue
		}

		// For numeric roots, use interval width reduction
		clonedDAG := p.dag.Clone()
		inputNode, _ := clonedDAG.GetNode(node.ID)

		mid := node.Interval.Midpoint()
		if math.IsInf(mid, 0) || math.IsNaN(mid) {
			mid = 0
		}
		inputNode.Interval = distribution.Point(mid)

		clonedProp := NewIntervalPropagator(clonedDAG)
		clonedProp.PropagateIntervals()

		clonedRoot := clonedDAG.GetRoot()
		newWidth := clonedRoot.Interval.Width()
		if math.IsInf(newWidth, 0) || math.IsNaN(newWidth) {
			newWidth = 1e12
		}

		// Sensitivity is the reduction in interval width
		reduction := baseWidth - newWidth
		if reduction < 0 {
			reduction = 0
		}
		sensitivity[node.ID] = reduction / baseWidth
	}

	return sensitivity
}

// BoolIntervalResult holds the result of boolean interval propagation.
type BoolIntervalResult struct {
	BoolInterval distribution.BoolInterval
	Confidence   float64 // Probability of the most likely outcome (if known)
}

// EvaluateBooleanExpr evaluates a boolean expression with interval arithmetic.
func (p *IntervalPropagator) EvaluateBooleanExpr(body string, intervals map[string]distribution.Interval) BoolIntervalResult {
	program, err := expr.Compile(body, expr.AllowUndefinedVariables())
	if err != nil {
		return BoolIntervalResult{BoolInterval: distribution.BoolIntervalUnknown}
	}

	result := p.evaluateASTBool(program.Node(), intervals)
	return result
}

func (p *IntervalPropagator) evaluateASTBool(node ast.Node, intervals map[string]distribution.Interval) BoolIntervalResult {
	switch n := node.(type) {
	case *ast.BoolNode:
		if n.Value {
			return BoolIntervalResult{BoolInterval: distribution.BoolIntervalTrue, Confidence: 1.0}
		}
		return BoolIntervalResult{BoolInterval: distribution.BoolIntervalFalse, Confidence: 1.0}

	case *ast.BinaryNode:
		switch n.Operator {
		case "<":
			left, _ := p.evaluateASTInterval(n.Left, intervals)
			right, _ := p.evaluateASTInterval(n.Right, intervals)
			return BoolIntervalResult{BoolInterval: left.Less(right)}

		case "<=":
			left, _ := p.evaluateASTInterval(n.Left, intervals)
			right, _ := p.evaluateASTInterval(n.Right, intervals)
			return BoolIntervalResult{BoolInterval: left.LessEq(right)}

		case ">":
			left, _ := p.evaluateASTInterval(n.Left, intervals)
			right, _ := p.evaluateASTInterval(n.Right, intervals)
			return BoolIntervalResult{BoolInterval: left.Greater(right)}

		case ">=":
			left, _ := p.evaluateASTInterval(n.Left, intervals)
			right, _ := p.evaluateASTInterval(n.Right, intervals)
			return BoolIntervalResult{BoolInterval: left.GreaterEq(right)}

		case "==":
			left, _ := p.evaluateASTInterval(n.Left, intervals)
			right, _ := p.evaluateASTInterval(n.Right, intervals)
			return BoolIntervalResult{BoolInterval: left.Equal(right)}

		case "!=":
			left, _ := p.evaluateASTInterval(n.Left, intervals)
			right, _ := p.evaluateASTInterval(n.Right, intervals)
			return BoolIntervalResult{BoolInterval: left.NotEqual(right)}

		case "&&", "and":
			left := p.evaluateASTBool(n.Left, intervals)
			right := p.evaluateASTBool(n.Right, intervals)
			return BoolIntervalResult{BoolInterval: left.BoolInterval.And(right.BoolInterval)}

		case "||", "or":
			left := p.evaluateASTBool(n.Left, intervals)
			right := p.evaluateASTBool(n.Right, intervals)
			return BoolIntervalResult{BoolInterval: left.BoolInterval.Or(right.BoolInterval)}
		}

	case *ast.UnaryNode:
		if n.Operator == "!" || n.Operator == "not" {
			operand := p.evaluateASTBool(n.Node, intervals)
			return BoolIntervalResult{BoolInterval: operand.BoolInterval.Not()}
		}
	}

	return BoolIntervalResult{BoolInterval: distribution.BoolIntervalUnknown}
}
