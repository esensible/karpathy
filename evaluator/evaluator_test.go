package evaluator

import (
	"math/rand"
	"testing"

	"github.com/karpathy/dag-evaluator/distribution"
	"github.com/karpathy/dag-evaluator/schema"
	"github.com/karpathy/dag-evaluator/voi"
)

func TestSimpleComparison(t *testing.T) {
	schemaJSON := `{
		"schema_version": "1.2",
		"name": "SimpleTest",
		"description": "Test A + B > 100",
		"root_node_id": "result",
		"enums": {},
		"parameters": {
			"threshold": {"description": "Threshold", "value": 100}
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
			"total": {
				"description": "Sum",
				"output_type": "float",
				"expression": {"expression_type": "EXPR", "body": "a + b"}
			},
			"result": {
				"description": "Result",
				"output_type": "bool",
				"expression": {"expression_type": "EXPR", "body": "total > threshold"}
			}
		}
	}`

	s, err := schema.LoadFromBytes([]byte(schemaJSON))
	if err != nil {
		t.Fatalf("Failed to load schema: %v", err)
	}

	cfg := Config{
		NumSamples:    500,
		NumVoISamples: 50,
		UtilityFunc:   voi.UtilityConfidence,
		CostWeight:    1.0,
		MaxIterations: 10,
		RNG:           rand.New(rand.NewSource(42)),
		Objective: voi.ResolutionObjective{
			Type:  voi.ResolveBoolean,
			Alpha: 0.95,
		},
	}

	eval, err := New(s, cfg)
	if err != nil {
		t.Fatalf("Failed to create evaluator: %v", err)
	}

	if err := eval.Initialize(); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Check initial confidence (should be around 0.5 since priors are symmetric)
	conf := eval.GetCurrentConfidence()
	t.Logf("Initial confidence: %.2f%%", conf*100)

	// Get VoI scores
	scores := eval.GetAllScores()
	if len(scores) != 2 {
		t.Errorf("Expected 2 scores, got %d", len(scores))
	}

	for _, score := range scores {
		t.Logf("Input %s: VoI=%.4f, Cost=%.2f, Net=%.4f",
			score.InputID, score.VoI, score.Cost, score.NetScore)
	}

	// Acquire both inputs with known values
	if err := eval.AcquireInput("a", distribution.NewFloatValue(60)); err != nil {
		t.Fatalf("Failed to acquire a: %v", err)
	}
	t.Logf("After acquiring a=60: confidence=%.2f%%", eval.GetCurrentConfidence()*100)

	if err := eval.AcquireInput("b", distribution.NewFloatValue(50)); err != nil {
		t.Fatalf("Failed to acquire b: %v", err)
	}
	t.Logf("After acquiring b=50: confidence=%.2f%%", eval.GetCurrentConfidence()*100)

	// Should be resolved now (60 + 50 = 110 > 100)
	result := eval.GetResult()
	if !result.IsResolved {
		t.Errorf("Expected resolved, got unresolved")
	}

	// Check the result value
	if result.Value.Type != distribution.ValueTypeBool {
		t.Errorf("Expected bool result, got %v", result.Value.Type)
	}
	if !result.Value.Bool {
		t.Errorf("Expected true (60+50=110 > 100), got false")
	}

	t.Logf("Final result: %s (cost: %.2f, steps: %d)",
		result.Value.AsString(), result.TotalCost, result.Steps)
}

func TestVoIRanking(t *testing.T) {
	// Test that VoI correctly ranks inputs by their value
	schemaJSON := `{
		"schema_version": "1.2",
		"name": "VoIRankingTest",
		"description": "Test VoI ranking with different costs",
		"root_node_id": "result",
		"enums": {},
		"parameters": {},
		"inputs": {
			"cheap": {
				"description": "Cheap input with narrow prior",
				"question": "Cheap?",
				"label": "Cheap",
				"output_type": "float",
				"acquisition_cost": 0.5,
				"prior": {"type": "uniform", "params": {"min": 40, "max": 60}}
			},
			"expensive": {
				"description": "Expensive input with wide prior",
				"question": "Expensive?",
				"label": "Expensive",
				"output_type": "float",
				"acquisition_cost": 10.0,
				"prior": {"type": "uniform", "params": {"min": 0, "max": 100}}
			}
		},
		"expressions": {
			"total": {
				"description": "Sum",
				"output_type": "float",
				"expression": {"expression_type": "EXPR", "body": "cheap + expensive"}
			},
			"result": {
				"description": "Result",
				"output_type": "bool",
				"expression": {"expression_type": "EXPR", "body": "total > 100"}
			}
		}
	}`

	s, err := schema.LoadFromBytes([]byte(schemaJSON))
	if err != nil {
		t.Fatalf("Failed to load schema: %v", err)
	}

	cfg := DefaultConfig()
	cfg.RNG = rand.New(rand.NewSource(42))

	eval, err := New(s, cfg)
	if err != nil {
		t.Fatalf("Failed to create evaluator: %v", err)
	}

	if err := eval.Initialize(); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	scores := eval.GetAllScores()
	for _, score := range scores {
		t.Logf("Input %s: VoI=%.4f, Cost=%.2f, Net=%.4f, Sensitivity=%.4f",
			score.InputID, score.VoI, score.Cost, score.NetScore, score.Sensitivity)
	}

	// The expensive input should have higher VoI (wider prior = more uncertainty reduction)
	// but may have lower net score due to cost
}

func TestDecisionTable(t *testing.T) {
	schemaJSON := `{
		"schema_version": "1.2",
		"name": "DecisionTableTest",
		"description": "Test decision table evaluation",
		"root_node_id": "category",
		"enums": {
			"Category": ["Low", "Medium", "High"]
		},
		"parameters": {},
		"inputs": {
			"score": {
				"description": "Score value",
				"question": "Enter score:",
				"label": "Score",
				"output_type": "float",
				"acquisition_cost": 1.0,
				"prior": {"type": "uniform", "params": {"min": 0, "max": 100}}
			}
		},
		"expressions": {
			"category": {
				"description": "Category based on score",
				"output_type": "Category",
				"expression": {
					"expression_type": "DECISION_TABLE",
					"table": {
						"hit_policy": "FIRST",
						"rules": [
							{"when": "score < 30", "then": "Category.Low"},
							{"when": "score < 70", "then": "Category.Medium"},
							{"when": "true", "then": "Category.High"}
						]
					}
				}
			}
		}
	}`

	s, err := schema.LoadFromBytes([]byte(schemaJSON))
	if err != nil {
		t.Fatalf("Failed to load schema: %v", err)
	}

	cfg := DefaultConfig()
	cfg.RNG = rand.New(rand.NewSource(42))
	cfg.Objective = voi.ResolutionObjective{
		Type:  voi.ResolveBoolean,
		Alpha: 0.8,
	}

	eval, err := New(s, cfg)
	if err != nil {
		t.Fatalf("Failed to create evaluator: %v", err)
	}

	if err := eval.Initialize(); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Test with score = 50 (should be Medium)
	if err := eval.AcquireInput("score", distribution.NewFloatValue(50)); err != nil {
		t.Fatalf("Failed to acquire score: %v", err)
	}

	result := eval.GetResult()
	t.Logf("Result: %s", result.Value.AsString())

	if result.Value.String != "Medium" {
		t.Errorf("Expected Medium, got %s", result.Value.String)
	}
}

func TestConditionalExpression(t *testing.T) {
	schemaJSON := `{
		"schema_version": "1.2",
		"name": "ConditionalTest",
		"description": "Test ternary conditional",
		"root_node_id": "result",
		"enums": {},
		"parameters": {},
		"inputs": {
			"condition": {
				"description": "Condition",
				"question": "Condition?",
				"label": "Condition",
				"output_type": "bool",
				"acquisition_cost": 1.0,
				"prior": {"type": "bernoulli", "params": {"p": 0.5}}
			},
			"a": {
				"description": "Value A",
				"question": "A?",
				"label": "A",
				"output_type": "float",
				"acquisition_cost": 2.0,
				"prior": {"type": "uniform", "params": {"min": 0, "max": 50}}
			},
			"b": {
				"description": "Value B",
				"question": "B?",
				"label": "B",
				"output_type": "float",
				"acquisition_cost": 2.0,
				"prior": {"type": "uniform", "params": {"min": 50, "max": 100}}
			}
		},
		"expressions": {
			"selected": {
				"description": "Selected value based on condition",
				"output_type": "float",
				"expression": {"expression_type": "EXPR", "body": "condition ? a : b"}
			},
			"result": {
				"description": "Whether selected > 30",
				"output_type": "bool",
				"expression": {"expression_type": "EXPR", "body": "selected > 30"}
			}
		}
	}`

	s, err := schema.LoadFromBytes([]byte(schemaJSON))
	if err != nil {
		t.Fatalf("Failed to load schema: %v", err)
	}

	cfg := DefaultConfig()
	cfg.RNG = rand.New(rand.NewSource(42))

	eval, err := New(s, cfg)
	if err != nil {
		t.Fatalf("Failed to create evaluator: %v", err)
	}

	if err := eval.Initialize(); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// First, acquire the condition
	if err := eval.AcquireInput("condition", distribution.NewBoolValue(true)); err != nil {
		t.Fatalf("Failed to acquire condition: %v", err)
	}

	// Now only 'a' should matter (not 'b')
	scores := eval.GetAllScores()
	t.Logf("After condition=true:")
	for _, score := range scores {
		t.Logf("  %s: VoI=%.4f, Net=%.4f", score.InputID, score.VoI, score.NetScore)
	}

	// Acquire 'a'
	if err := eval.AcquireInput("a", distribution.NewFloatValue(40)); err != nil {
		t.Fatalf("Failed to acquire a: %v", err)
	}

	result := eval.GetResult()
	t.Logf("Result: %s (a=40, selected=40, 40>30=%t)", result.Value.AsString(), result.Value.Bool)

	if !result.Value.Bool {
		t.Errorf("Expected true (40 > 30), got false")
	}
}

func TestAutoEvaluation(t *testing.T) {
	schemaJSON := `{
		"schema_version": "1.2",
		"name": "AutoTest",
		"description": "Test automatic evaluation",
		"root_node_id": "result",
		"enums": {},
		"parameters": {},
		"inputs": {
			"x": {
				"description": "X",
				"question": "X?",
				"label": "X",
				"output_type": "float",
				"acquisition_cost": 0.1,
				"prior": {"type": "uniform", "params": {"min": 0, "max": 100}}
			}
		},
		"expressions": {
			"result": {
				"description": "X > 50",
				"output_type": "bool",
				"expression": {"expression_type": "EXPR", "body": "x > 50"}
			}
		}
	}`

	s, err := schema.LoadFromBytes([]byte(schemaJSON))
	if err != nil {
		t.Fatalf("Failed to load schema: %v", err)
	}

	cfg := DefaultConfig()
	cfg.RNG = rand.New(rand.NewSource(42))
	cfg.CostWeight = 0.01 // Lower cost weight so VoI dominates

	eval, err := New(s, cfg)
	if err != nil {
		t.Fatalf("Failed to create evaluator: %v", err)
	}

	// Run with fixed provider
	provider := FixedProvider(map[string]interface{}{
		"x": 75.0,
	})

	result, err := eval.RunAutomatic(provider, 0)
	if err != nil {
		t.Fatalf("Failed to run automatic: %v", err)
	}

	t.Logf("Auto result: %s (cost: %.2f, steps: %d)",
		result.Value.AsString(), result.TotalCost, result.Steps)

	if !result.Value.Bool {
		t.Errorf("Expected true (75 > 50), got false")
	}
}
