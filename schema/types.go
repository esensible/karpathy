// Package schema defines the types for loading DAG expression schemas from JSON.
package schema

import (
	"encoding/json"
	"fmt"
)

// Schema represents the top-level DAG expression schema.
type Schema struct {
	SchemaVersion string                  `json:"schema_version"`
	Name          string                  `json:"name"`
	Description   string                  `json:"description"`
	RootNodeID    string                  `json:"root_node_id"`
	Enums         map[string][]string     `json:"enums"`
	Parameters    map[string]Parameter    `json:"parameters"`
	Inputs        map[string]InputNode    `json:"inputs"`
	Expressions   map[string]Expression   `json:"expressions"`
}

// Parameter represents a constant value defined in the schema.
type Parameter struct {
	Description string      `json:"description"`
	Value       interface{} `json:"value"`
}

// InputNode represents a user-provided input variable.
type InputNode struct {
	Description    string                 `json:"description"`
	Question       string                 `json:"question"`
	Label          string                 `json:"label"`
	OutputType     OutputType             `json:"output_type"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	EmbeddingModel string                 `json:"embedding_model,omitempty"`
	// Cost fields for the evaluation framework
	AcquisitionCost float64      `json:"acquisition_cost,omitempty"`
	Prior           *PriorConfig `json:"prior,omitempty"`
}

// PriorConfig defines the prior distribution for an input.
type PriorConfig struct {
	Type   string                 `json:"type"` // "uniform", "normal", "truncated_normal", "categorical", etc.
	Params map[string]interface{} `json:"params"`
}

// OutputType can be a string (primitive or enum) or a map (complex type).
type OutputType struct {
	Primitive  string            // For simple types: "str", "int", "float", "bool"
	Complex    map[string]string // For object types: {"field1": "type1", "field2": "type2"}
	IsComplex  bool
}

// UnmarshalJSON handles both string and object output types.
func (o *OutputType) UnmarshalJSON(data []byte) error {
	// Try string first
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		o.Primitive = s
		o.IsComplex = false
		return nil
	}

	// Try object
	var m map[string]string
	if err := json.Unmarshal(data, &m); err == nil {
		o.Complex = m
		o.IsComplex = true
		return nil
	}

	return fmt.Errorf("output_type must be a string or object, got: %s", string(data))
}

// MarshalJSON serializes OutputType back to JSON.
func (o OutputType) MarshalJSON() ([]byte, error) {
	if o.IsComplex {
		return json.Marshal(o.Complex)
	}
	return json.Marshal(o.Primitive)
}

// Expression represents a computed expression node.
type Expression struct {
	Description string         `json:"description"`
	OutputType  OutputType     `json:"output_type"`
	Expression  ExpressionBody `json:"expression"`
}

// ExpressionBody holds the expression type and body/table.
type ExpressionBody struct {
	ExpressionType string         `json:"expression_type"` // "EXPR" or "DECISION_TABLE"
	Body           string         `json:"body,omitempty"`
	Table          *DecisionTable `json:"table,omitempty"`
}

// DecisionTable represents a decision table expression.
type DecisionTable struct {
	HitPolicy string `json:"hit_policy"` // "FIRST" or "COLLECT"
	Rules     []Rule `json:"rules"`
}

// Rule represents a single rule in a decision table.
type Rule struct {
	Description string      `json:"description,omitempty"`
	When        string      `json:"when"`
	Then        interface{} `json:"then"` // Can be string, number, or expression
}

// ThenAsString returns the Then value as a string expression.
func (r Rule) ThenAsString() string {
	switch v := r.Then.(type) {
	case string:
		return v
	case float64:
		return fmt.Sprintf("%v", v)
	case int:
		return fmt.Sprintf("%d", v)
	case bool:
		return fmt.Sprintf("%t", v)
	default:
		// For complex objects, marshal to JSON-like expression syntax
		if m, ok := v.(map[string]interface{}); ok {
			return mapToExpr(m)
		}
		return fmt.Sprintf("%v", v)
	}
}

func mapToExpr(m map[string]interface{}) string {
	result := "{"
	first := true
	for k, v := range m {
		if !first {
			result += ", "
		}
		first = false
		switch val := v.(type) {
		case string:
			result += fmt.Sprintf("%s: %s", k, val)
		case float64:
			result += fmt.Sprintf("%s: %v", k, val)
		default:
			result += fmt.Sprintf("%s: %v", k, val)
		}
	}
	result += "}"
	return result
}

// NodeType represents the type of a node in the DAG.
type NodeType int

const (
	NodeTypeInput NodeType = iota
	NodeTypeExpression
	NodeTypeParameter
)

// String returns the string representation of NodeType.
func (n NodeType) String() string {
	switch n {
	case NodeTypeInput:
		return "input"
	case NodeTypeExpression:
		return "expression"
	case NodeTypeParameter:
		return "parameter"
	default:
		return "unknown"
	}
}

// DataType represents the data type of a value.
type DataType int

const (
	DataTypeUnknown DataType = iota
	DataTypeInt
	DataTypeFloat
	DataTypeString
	DataTypeBool
	DataTypeEnum
	DataTypeObject
	DataTypeArray
)

// String returns the string representation of DataType.
func (d DataType) String() string {
	switch d {
	case DataTypeInt:
		return "int"
	case DataTypeFloat:
		return "float"
	case DataTypeString:
		return "string"
	case DataTypeBool:
		return "bool"
	case DataTypeEnum:
		return "enum"
	case DataTypeObject:
		return "object"
	case DataTypeArray:
		return "array"
	default:
		return "unknown"
	}
}

// ParseDataType converts a string type name to DataType.
func ParseDataType(s string) DataType {
	switch s {
	case "int":
		return DataTypeInt
	case "float":
		return DataTypeFloat
	case "str", "string":
		return DataTypeString
	case "bool", "boolean":
		return DataTypeBool
	default:
		// Could be an enum type
		return DataTypeEnum
	}
}
