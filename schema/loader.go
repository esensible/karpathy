package schema

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Load reads and parses a schema from a JSON file.
func Load(path string) (*Schema, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open schema file: %w", err)
	}
	defer f.Close()
	return LoadFromReader(f)
}

// LoadFromReader parses a schema from a reader.
func LoadFromReader(r io.Reader) (*Schema, error) {
	var schema Schema
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&schema); err != nil {
		return nil, fmt.Errorf("failed to parse schema JSON: %w", err)
	}
	if err := schema.Validate(); err != nil {
		return nil, fmt.Errorf("schema validation failed: %w", err)
	}
	return &schema, nil
}

// LoadFromBytes parses a schema from JSON bytes.
func LoadFromBytes(data []byte) (*Schema, error) {
	var schema Schema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("failed to parse schema JSON: %w", err)
	}
	if err := schema.Validate(); err != nil {
		return nil, fmt.Errorf("schema validation failed: %w", err)
	}
	return &schema, nil
}

// Validate checks the schema for consistency and correctness.
func (s *Schema) Validate() error {
	if s.Name == "" {
		return fmt.Errorf("schema name is required")
	}
	if s.RootNodeID == "" {
		return fmt.Errorf("root_node_id is required")
	}

	// Check that root node exists
	if _, ok := s.Expressions[s.RootNodeID]; !ok {
		if _, ok := s.Inputs[s.RootNodeID]; !ok {
			return fmt.Errorf("root_node_id '%s' not found in expressions or inputs", s.RootNodeID)
		}
	}

	// Validate enum definitions
	for name, values := range s.Enums {
		if len(values) == 0 {
			return fmt.Errorf("enum '%s' has no values", name)
		}
		for _, v := range values {
			if containsSpace(v) {
				return fmt.Errorf("enum value '%s' in '%s' contains spaces", v, name)
			}
		}
	}

	// Validate input types reference valid enums
	for id, input := range s.Inputs {
		if !input.OutputType.IsComplex {
			if err := s.validateType(input.OutputType.Primitive, id); err != nil {
				return err
			}
		}
	}

	// Validate expression types
	for id, expr := range s.Expressions {
		if expr.Expression.ExpressionType != "EXPR" && expr.Expression.ExpressionType != "DECISION_TABLE" {
			return fmt.Errorf("expression '%s' has invalid expression_type: %s", id, expr.Expression.ExpressionType)
		}
		if expr.Expression.ExpressionType == "EXPR" && expr.Expression.Body == "" {
			return fmt.Errorf("expression '%s' has EXPR type but no body", id)
		}
		if expr.Expression.ExpressionType == "DECISION_TABLE" && expr.Expression.Table == nil {
			return fmt.Errorf("expression '%s' has DECISION_TABLE type but no table", id)
		}
	}

	return nil
}

func (s *Schema) validateType(typeName string, nodeID string) error {
	// Check if it's a primitive type
	switch typeName {
	case "str", "string", "int", "float", "bool", "boolean":
		return nil
	}
	// Check if it's a defined enum
	if _, ok := s.Enums[typeName]; ok {
		return nil
	}
	return fmt.Errorf("node '%s' references undefined type '%s'", nodeID, typeName)
}

func containsSpace(s string) bool {
	for _, c := range s {
		if c == ' ' {
			return true
		}
	}
	return false
}

// GetAllNodeIDs returns all node IDs (inputs + expressions).
func (s *Schema) GetAllNodeIDs() []string {
	ids := make([]string, 0, len(s.Inputs)+len(s.Expressions))
	for id := range s.Inputs {
		ids = append(ids, id)
	}
	for id := range s.Expressions {
		ids = append(ids, id)
	}
	return ids
}

// GetNodeType returns the type of a node given its ID.
func (s *Schema) GetNodeType(id string) (NodeType, bool) {
	if _, ok := s.Inputs[id]; ok {
		return NodeTypeInput, true
	}
	if _, ok := s.Expressions[id]; ok {
		return NodeTypeExpression, true
	}
	if _, ok := s.Parameters[id]; ok {
		return NodeTypeParameter, true
	}
	return 0, false
}

// IsEnumType checks if a type name is a defined enum.
func (s *Schema) IsEnumType(typeName string) bool {
	_, ok := s.Enums[typeName]
	return ok
}

// GetEnumValues returns the values for an enum type.
func (s *Schema) GetEnumValues(enumName string) ([]string, bool) {
	values, ok := s.Enums[enumName]
	return values, ok
}
