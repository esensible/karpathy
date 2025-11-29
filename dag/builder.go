package dag

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/expr-lang/expr"
	"github.com/expr-lang/expr/ast"
	"github.com/karpathy/dag-evaluator/distribution"
	"github.com/karpathy/dag-evaluator/schema"
)

// Builder constructs a DAG from a schema.
type Builder struct {
	schema *schema.Schema
	dag    *DAG
}

// NewBuilder creates a new DAG builder.
func NewBuilder(s *schema.Schema) *Builder {
	return &Builder{
		schema: s,
		dag: &DAG{
			Schema:     s,
			Nodes:      make(map[string]*Node),
			RootNodeID: s.RootNodeID,
			Enums:      s.Enums,
		},
	}
}

// Build constructs the complete DAG from the schema.
func (b *Builder) Build() (*DAG, error) {
	// Create parameter nodes
	for id, param := range b.schema.Parameters {
		if err := b.createParameterNode(id, param); err != nil {
			return nil, fmt.Errorf("failed to create parameter node '%s': %w", id, err)
		}
	}

	// Create input nodes
	for id, input := range b.schema.Inputs {
		if err := b.createInputNode(id, input); err != nil {
			return nil, fmt.Errorf("failed to create input node '%s': %w", id, err)
		}
	}

	// Create expression nodes
	for id, expr := range b.schema.Expressions {
		if err := b.createExpressionNode(id, expr); err != nil {
			return nil, fmt.Errorf("failed to create expression node '%s': %w", id, err)
		}
	}

	// Extract dependencies
	if err := b.extractDependencies(); err != nil {
		return nil, fmt.Errorf("failed to extract dependencies: %w", err)
	}

	// Build topological order
	if err := b.buildTopologicalOrder(); err != nil {
		return nil, fmt.Errorf("failed to build topological order: %w", err)
	}

	return b.dag, nil
}

func (b *Builder) createParameterNode(id string, param schema.Parameter) error {
	node := NewNode(id, schema.NodeTypeParameter)
	node.Description = param.Description
	node.Parameter = &ParameterInfo{Value: param.Value}

	// Parameters are always resolved
	node.IsResolved = true
	switch v := param.Value.(type) {
	case float64:
		node.Value = distribution.NewFloatValue(v)
		node.DataType = schema.DataTypeFloat
	case int:
		node.Value = distribution.NewIntValue(int64(v))
		node.DataType = schema.DataTypeInt
	case int64:
		node.Value = distribution.NewIntValue(v)
		node.DataType = schema.DataTypeInt
	case bool:
		node.Value = distribution.NewBoolValue(v)
		node.DataType = schema.DataTypeBool
	case string:
		node.Value = distribution.NewStringValue(v)
		node.DataType = schema.DataTypeString
	default:
		node.Value = distribution.NewFloatValue(0)
		node.DataType = schema.DataTypeUnknown
	}
	node.Distribution = distribution.NewPointDistribution(node.Value)

	b.dag.Nodes[id] = node
	return nil
}

func (b *Builder) createInputNode(id string, input schema.InputNode) error {
	node := NewNode(id, schema.NodeTypeInput)
	node.Description = input.Description
	node.DataType = b.resolveDataType(input.OutputType)

	// Set up input info
	cost := input.AcquisitionCost
	if cost == 0 {
		cost = 1.0 // Default cost
	}

	inputInfo := &InputInfo{
		Question:        input.Question,
		Label:           input.Label,
		AcquisitionCost: cost,
	}

	// Determine prior distribution based on type and configuration
	prior := b.createPriorDistribution(input, node.DataType)
	inputInfo.Prior = prior
	node.Distribution = prior
	node.Interval = distribution.IntervalFromDistribution(prior)

	// Handle enum types
	if !input.OutputType.IsComplex {
		if values, ok := b.schema.Enums[input.OutputType.Primitive]; ok {
			inputInfo.EnumType = input.OutputType.Primitive
			inputInfo.EnumValues = values
		}
	}

	node.Input = inputInfo
	b.dag.Nodes[id] = node
	return nil
}

func (b *Builder) createPriorDistribution(input schema.InputNode, dataType schema.DataType) distribution.Distribution {
	// Check for explicit prior configuration
	if input.Prior != nil {
		return b.createDistributionFromConfig(input.Prior, dataType, input.OutputType)
	}

	// Default priors based on type
	switch dataType {
	case schema.DataTypeFloat:
		// Default: wide uniform
		return distribution.NewUniformDistribution(-1e6, 1e6)
	case schema.DataTypeInt:
		return distribution.NewUniformDistribution(-1e6, 1e6)
	case schema.DataTypeBool:
		return distribution.NewBoolDistribution(0.5)
	case schema.DataTypeEnum:
		if values, ok := b.schema.Enums[input.OutputType.Primitive]; ok {
			return distribution.NewCategoricalDistribution(values, input.OutputType.Primitive)
		}
		return distribution.NewCategoricalDistribution([]string{"unknown"}, "")
	case schema.DataTypeString:
		return distribution.NewCategoricalDistribution([]string{"unknown"}, "")
	default:
		return distribution.NewUniformDistribution(-1e6, 1e6)
	}
}

func (b *Builder) createDistributionFromConfig(cfg *schema.PriorConfig, dataType schema.DataType, outputType schema.OutputType) distribution.Distribution {
	switch cfg.Type {
	case "uniform":
		min, _ := cfg.Params["min"].(float64)
		max, _ := cfg.Params["max"].(float64)
		if max == 0 && min == 0 {
			max = 1e6
			min = -1e6
		}
		return distribution.NewUniformDistribution(min, max)

	case "normal", "gaussian":
		mu, _ := cfg.Params["mu"].(float64)
		sigma, _ := cfg.Params["sigma"].(float64)
		if sigma == 0 {
			sigma = 1.0
		}
		return distribution.NewNormalDistribution(mu, sigma)

	case "truncated_normal":
		mu, _ := cfg.Params["mu"].(float64)
		sigma, _ := cfg.Params["sigma"].(float64)
		lower, _ := cfg.Params["lower"].(float64)
		upper, _ := cfg.Params["upper"].(float64)
		if sigma == 0 {
			sigma = 1.0
		}
		return distribution.NewTruncatedNormalDistribution(mu, sigma, lower, upper)

	case "bernoulli", "bool":
		p, ok := cfg.Params["p"].(float64)
		if !ok {
			p = 0.5
		}
		return distribution.NewBoolDistribution(p)

	case "categorical":
		if values, ok := cfg.Params["values"].([]interface{}); ok {
			strValues := make([]string, len(values))
			for i, v := range values {
				strValues[i] = fmt.Sprintf("%v", v)
			}
			enumType := ""
			if !outputType.IsComplex {
				if _, ok := b.schema.Enums[outputType.Primitive]; ok {
					enumType = outputType.Primitive
				}
			}
			if probs, ok := cfg.Params["probs"].([]interface{}); ok {
				floatProbs := make([]float64, len(probs))
				for i, p := range probs {
					floatProbs[i], _ = p.(float64)
				}
				return distribution.NewCategoricalDistributionWithProbs(strValues, floatProbs, enumType)
			}
			return distribution.NewCategoricalDistribution(strValues, enumType)
		}
		return distribution.NewCategoricalDistribution([]string{"unknown"}, "")

	default:
		// Fall back to default
		return distribution.NewUniformDistribution(-1e6, 1e6)
	}
}

func (b *Builder) createExpressionNode(id string, exprDef schema.Expression) error {
	node := NewNode(id, schema.NodeTypeExpression)
	node.Description = exprDef.Description
	node.DataType = b.resolveDataType(exprDef.OutputType)

	exprInfo := &ExpressionInfo{
		ExprType: exprDef.Expression.ExpressionType,
	}

	if exprDef.Expression.ExpressionType == "EXPR" {
		exprInfo.Body = exprDef.Expression.Body
	} else if exprDef.Expression.ExpressionType == "DECISION_TABLE" {
		table := exprDef.Expression.Table
		tableInfo := &DecisionTableInfo{
			HitPolicy: table.HitPolicy,
			Rules:     make([]RuleInfo, len(table.Rules)),
		}
		for i, rule := range table.Rules {
			tableInfo.Rules[i] = RuleInfo{
				Description: rule.Description,
				WhenExpr:    rule.When,
				ThenExpr:    rule.ThenAsString(),
			}
		}
		exprInfo.Table = tableInfo
	}

	node.Expression = exprInfo
	b.dag.Nodes[id] = node
	return nil
}

func (b *Builder) resolveDataType(outputType schema.OutputType) schema.DataType {
	if outputType.IsComplex {
		return schema.DataTypeObject
	}
	switch outputType.Primitive {
	case "int":
		return schema.DataTypeInt
	case "float":
		return schema.DataTypeFloat
	case "str", "string":
		return schema.DataTypeString
	case "bool", "boolean":
		return schema.DataTypeBool
	default:
		// Check if it's an enum
		if _, ok := b.schema.Enums[outputType.Primitive]; ok {
			return schema.DataTypeEnum
		}
		return schema.DataTypeUnknown
	}
}

func (b *Builder) extractDependencies() error {
	// Build a set of all valid node and parameter names
	validNames := make(map[string]bool)
	for id := range b.dag.Nodes {
		validNames[id] = true
	}
	// Also add enum type names (for Enum.Value access)
	for enumName := range b.schema.Enums {
		validNames[enumName] = true
	}

	for id, node := range b.dag.Nodes {
		if node.IsExpression() {
			deps, err := b.extractExpressionDependencies(node.Expression)
			if err != nil {
				return fmt.Errorf("node '%s': %w", id, err)
			}
			// Filter to only valid node names
			for _, dep := range deps {
				if validNames[dep] && dep != id {
					// Check it's an actual node (not just an enum type)
					if _, ok := b.dag.Nodes[dep]; ok {
						node.Dependencies = append(node.Dependencies, dep)
						// Add reverse dependency
						if depNode, ok := b.dag.Nodes[dep]; ok {
							depNode.Dependents = append(depNode.Dependents, id)
						}
					}
				}
			}
		}
	}
	return nil
}

func (b *Builder) extractExpressionDependencies(exprInfo *ExpressionInfo) ([]string, error) {
	deps := make(map[string]bool)

	if exprInfo.ExprType == "EXPR" {
		extracted, err := extractIdentifiersFromExpr(exprInfo.Body)
		if err != nil {
			return nil, err
		}
		for _, id := range extracted {
			deps[id] = true
		}
	} else if exprInfo.Table != nil {
		for _, rule := range exprInfo.Table.Rules {
			whenDeps, err := extractIdentifiersFromExpr(rule.WhenExpr)
			if err != nil {
				return nil, err
			}
			for _, id := range whenDeps {
				deps[id] = true
			}
			thenDeps, err := extractIdentifiersFromExpr(rule.ThenExpr)
			if err != nil {
				return nil, err
			}
			for _, id := range thenDeps {
				deps[id] = true
			}
		}
	}

	result := make([]string, 0, len(deps))
	for dep := range deps {
		result = append(result, dep)
	}
	return result, nil
}

// extractIdentifiersFromExpr extracts variable identifiers from an expression string.
func extractIdentifiersFromExpr(exprStr string) ([]string, error) {
	if exprStr == "" {
		return nil, nil
	}

	// Try to parse with expr-lang to get AST
	program, err := expr.Compile(exprStr, expr.AllowUndefinedVariables())
	if err != nil {
		// Fall back to regex-based extraction
		return extractIdentifiersRegex(exprStr), nil
	}

	// Walk the AST to find identifiers
	identifiers := make(map[string]bool)
	walkAST(program.Node(), identifiers)

	result := make([]string, 0, len(identifiers))
	for id := range identifiers {
		result = append(result, id)
	}
	return result, nil
}

func walkAST(node ast.Node, identifiers map[string]bool) {
	if node == nil {
		return
	}

	switch n := node.(type) {
	case *ast.IdentifierNode:
		identifiers[n.Value] = true
	case *ast.MemberNode:
		// For member access like Enum.Value, we want the base identifier
		if ident, ok := n.Node.(*ast.IdentifierNode); ok {
			identifiers[ident.Value] = true
		}
		walkAST(n.Node, identifiers)
	case *ast.CallNode:
		walkAST(n.Callee, identifiers)
		for _, arg := range n.Arguments {
			walkAST(arg, identifiers)
		}
	case *ast.BinaryNode:
		walkAST(n.Left, identifiers)
		walkAST(n.Right, identifiers)
	case *ast.UnaryNode:
		walkAST(n.Node, identifiers)
	case *ast.ConditionalNode:
		walkAST(n.Cond, identifiers)
		walkAST(n.Exp1, identifiers)
		walkAST(n.Exp2, identifiers)
	case *ast.ArrayNode:
		for _, elem := range n.Nodes {
			walkAST(elem, identifiers)
		}
	case *ast.MapNode:
		for _, pair := range n.Pairs {
			walkAST(pair, identifiers)
		}
	}
}

// extractIdentifiersRegex is a fallback for when AST parsing fails.
func extractIdentifiersRegex(exprStr string) []string {
	// Match identifiers that look like variable names
	re := regexp.MustCompile(`\b([a-zA-Z_][a-zA-Z0-9_]*)\b`)
	matches := re.FindAllStringSubmatch(exprStr, -1)

	// Keywords and built-ins to exclude
	keywords := map[string]bool{
		"true": true, "false": true, "nil": true, "null": true,
		"if": true, "else": true, "and": true, "or": true, "not": true,
		"in": true, "matches": true, "contains": true,
		"len": true, "all": true, "any": true, "one": true, "none": true,
		"map": true, "filter": true, "count": true,
	}

	seen := make(map[string]bool)
	var result []string
	for _, match := range matches {
		id := match[1]
		if !keywords[id] && !seen[id] {
			// Skip if it looks like an enum value (PascalCase after a dot)
			if !strings.Contains(exprStr, "."+id) || !isUpperFirst(id) {
				seen[id] = true
				result = append(result, id)
			}
		}
	}
	return result
}

func isUpperFirst(s string) bool {
	if len(s) == 0 {
		return false
	}
	return s[0] >= 'A' && s[0] <= 'Z'
}

func (b *Builder) buildTopologicalOrder() error {
	// Kahn's algorithm
	inDegree := make(map[string]int)
	for id := range b.dag.Nodes {
		inDegree[id] = 0
	}
	for _, node := range b.dag.Nodes {
		for _, dep := range node.Dependencies {
			inDegree[node.ID]++
			_ = dep // used for counting
		}
	}

	// Initialize queue with nodes having no dependencies
	var queue []string
	for id, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, id)
		}
	}

	var order []string
	for len(queue) > 0 {
		// Pop from queue
		current := queue[0]
		queue = queue[1:]
		order = append(order, current)

		// Reduce in-degree of dependents
		node := b.dag.Nodes[current]
		for _, depID := range node.Dependents {
			inDegree[depID]--
			if inDegree[depID] == 0 {
				queue = append(queue, depID)
			}
		}
	}

	// Check for cycles
	if len(order) != len(b.dag.Nodes) {
		return fmt.Errorf("circular dependency detected")
	}

	b.dag.TopoOrder = order
	return nil
}

// BuildFromSchema is a convenience function to build a DAG from a schema.
func BuildFromSchema(s *schema.Schema) (*DAG, error) {
	builder := NewBuilder(s)
	return builder.Build()
}
