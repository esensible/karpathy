// Package dag provides the DAG representation of expression trees.
package dag

import (
	"github.com/karpathy/dag-evaluator/distribution"
	"github.com/karpathy/dag-evaluator/schema"
)

// Node represents a node in the expression DAG.
type Node struct {
	ID          string
	Type        schema.NodeType
	DataType    schema.DataType
	Description string

	// For input nodes
	Input *InputInfo

	// For expression nodes
	Expression *ExpressionInfo

	// For parameter nodes
	Parameter *ParameterInfo

	// Dependency tracking
	Dependencies []string // IDs of nodes this node depends on
	Dependents   []string // IDs of nodes that depend on this node

	// Current state
	Distribution distribution.Distribution
	Interval     distribution.Interval
	IsResolved   bool
	Value        distribution.Value
}

// InputInfo contains information about an input node.
type InputInfo struct {
	Question        string
	Label           string
	AcquisitionCost float64
	Prior           distribution.Distribution
	EnumType        string // Non-empty if this is an enum input
	EnumValues      []string
}

// ExpressionInfo contains information about an expression node.
type ExpressionInfo struct {
	ExprType string // "EXPR" or "DECISION_TABLE"
	Body     string
	Table    *DecisionTableInfo
}

// DecisionTableInfo contains parsed decision table information.
type DecisionTableInfo struct {
	HitPolicy string
	Rules     []RuleInfo
}

// RuleInfo contains information about a single rule.
type RuleInfo struct {
	Description string
	WhenExpr    string
	ThenExpr    string
}

// ParameterInfo contains information about a parameter node.
type ParameterInfo struct {
	Value interface{}
}

// DAG represents the complete expression DAG.
type DAG struct {
	Schema     *schema.Schema
	Nodes      map[string]*Node
	RootNodeID string

	// Topological ordering for evaluation
	TopoOrder []string

	// Enum definitions for quick lookup
	Enums map[string][]string
}

// NewNode creates a new node.
func NewNode(id string, nodeType schema.NodeType) *Node {
	return &Node{
		ID:           id,
		Type:         nodeType,
		Dependencies: make([]string, 0),
		Dependents:   make([]string, 0),
		IsResolved:   false,
	}
}

// SetResolved marks the node as resolved with a specific value.
func (n *Node) SetResolved(v distribution.Value) {
	n.Value = v
	n.IsResolved = true
	n.Distribution = distribution.NewPointDistribution(v)
	if f, ok := v.AsFloat(); ok {
		n.Interval = distribution.Point(f)
	}
}

// SetDistribution updates the node's distribution.
func (n *Node) SetDistribution(d distribution.Distribution) {
	n.Distribution = d
	n.Interval = distribution.IntervalFromDistribution(d)
	if d.IsPoint() {
		if v, ok := d.PointValue(); ok {
			n.Value = v
			n.IsResolved = true
		}
	}
}

// IsInput returns true if this is an input node.
func (n *Node) IsInput() bool {
	return n.Type == schema.NodeTypeInput
}

// IsExpression returns true if this is an expression node.
func (n *Node) IsExpression() bool {
	return n.Type == schema.NodeTypeExpression
}

// IsParameter returns true if this is a parameter node.
func (n *Node) IsParameter() bool {
	return n.Type == schema.NodeTypeParameter
}

// GetCost returns the acquisition cost of the node (0 for non-inputs).
func (n *Node) GetCost() float64 {
	if n.Input != nil {
		return n.Input.AcquisitionCost
	}
	return 0
}

// GetUnresolvedInputs returns the IDs of unresolved input dependencies.
func (d *DAG) GetUnresolvedInputs(nodeID string) []string {
	visited := make(map[string]bool)
	var result []string
	d.collectUnresolvedInputs(nodeID, visited, &result)
	return result
}

func (d *DAG) collectUnresolvedInputs(nodeID string, visited map[string]bool, result *[]string) {
	if visited[nodeID] {
		return
	}
	visited[nodeID] = true

	node, ok := d.Nodes[nodeID]
	if !ok {
		return
	}

	if node.IsInput() && !node.IsResolved {
		*result = append(*result, nodeID)
		return
	}

	for _, depID := range node.Dependencies {
		d.collectUnresolvedInputs(depID, visited, result)
	}
}

// GetNode returns a node by ID.
func (d *DAG) GetNode(id string) (*Node, bool) {
	node, ok := d.Nodes[id]
	return node, ok
}

// GetRoot returns the root node.
func (d *DAG) GetRoot() *Node {
	return d.Nodes[d.RootNodeID]
}

// AllInputsResolved returns true if all input nodes are resolved.
func (d *DAG) AllInputsResolved() bool {
	for _, node := range d.Nodes {
		if node.IsInput() && !node.IsResolved {
			return false
		}
	}
	return true
}

// GetAllInputNodes returns all input nodes.
func (d *DAG) GetAllInputNodes() []*Node {
	var inputs []*Node
	for _, node := range d.Nodes {
		if node.IsInput() {
			inputs = append(inputs, node)
		}
	}
	return inputs
}

// GetUnresolvedInputNodes returns all unresolved input nodes.
func (d *DAG) GetUnresolvedInputNodes() []*Node {
	var inputs []*Node
	for _, node := range d.Nodes {
		if node.IsInput() && !node.IsResolved {
			inputs = append(inputs, node)
		}
	}
	return inputs
}

// Clone creates a deep copy of the DAG.
func (d *DAG) Clone() *DAG {
	newDAG := &DAG{
		Schema:     d.Schema,
		Nodes:      make(map[string]*Node),
		RootNodeID: d.RootNodeID,
		TopoOrder:  make([]string, len(d.TopoOrder)),
		Enums:      d.Enums,
	}

	copy(newDAG.TopoOrder, d.TopoOrder)

	for id, node := range d.Nodes {
		newNode := &Node{
			ID:           node.ID,
			Type:         node.Type,
			DataType:     node.DataType,
			Description:  node.Description,
			Dependencies: make([]string, len(node.Dependencies)),
			Dependents:   make([]string, len(node.Dependents)),
			IsResolved:   node.IsResolved,
			Value:        node.Value,
			Interval:     node.Interval,
		}
		copy(newNode.Dependencies, node.Dependencies)
		copy(newNode.Dependents, node.Dependents)

		if node.Distribution != nil {
			newNode.Distribution = node.Distribution.Clone()
		}
		if node.Input != nil {
			newNode.Input = &InputInfo{
				Question:        node.Input.Question,
				Label:           node.Input.Label,
				AcquisitionCost: node.Input.AcquisitionCost,
				EnumType:        node.Input.EnumType,
				EnumValues:      node.Input.EnumValues,
			}
			if node.Input.Prior != nil {
				newNode.Input.Prior = node.Input.Prior.Clone()
			}
		}
		if node.Expression != nil {
			newNode.Expression = node.Expression // Immutable, can share
		}
		if node.Parameter != nil {
			newNode.Parameter = node.Parameter // Immutable, can share
		}
		newDAG.Nodes[id] = newNode
	}

	return newDAG
}
